'''
This file defines the training loop for the GCN model. It includes:
- the monotonicity loss function to enforce the health index to be monotone with respect to the damage states
- the train function that sets up the model, loads the datasets, and runs the training loop with optional validation
- a plot_HI function to visualize the learned health index against the states after training

'''

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "tools", "training", "intermediate_results_check", "results_analysis"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import datetime
import os

from GCN import DeepGraphCNN
from graph_dataset import features_GraphDataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric
from imagining_alghoritm import optimal_beta, MAX_DIST, U as _wcpdi_U
from config import (
    BETA_CONSTANT, DEFAULT_BATCH_SIZE, EPOCHS_PER_FOLD_GCN, GCN_MODELS_DIR, GRAPH_DATA_DIR, CROSS_VALIDATION_RESULTS_DIR,
    DEFAULT_FREQ_INDEX,
    DEFAULT_N_FEATURES, BASE_PANELS, TEST_PANEL,
    SENSOR_POSITIONS, SENSOR_PAIRS, DAMAGE_POINTS, PANEL_W, PANEL_H, VAL_PANELS, LR
)

def monotonicity_loss(values, state_ids, panel_ids):
    # values:    (batch_size, k) — one or more scalars per graph (1 x HI or 28 sub-HI)
    # state_ids: (batch_size,)   — true damage-state index of each graph
    # panel_ids: (batch_size,)   — panel identity of each graph (state_ids restart at 0 per panel)
   
    values = values.reshape(values.shape[0], -1)
    loss = values.sum() * 0.0
    for panel in panel_ids.unique():
        mask = panel_ids == panel
        if mask.sum() < 2:
            continue
        states = state_ids[mask]
        order = torch.argsort(states)
        sorted_states = states[order]
        sorted_values = values[mask][order]

        gaps = sorted_states[1:] - sorted_states[:-1]
        adjacent = gaps == 1
        if adjacent.sum() == 0:
            continue

        diff = sorted_values[1:] - sorted_values[:-1]
        diff = diff[adjacent]
        loss = loss + (((diff + 10.0) ** 2) - 100.0).mean(dim=1).sum() # mean over number of paths dimmension to make the value closer to the global loss
    return loss

def combined_path_loss(HI, out, state_ids, panel_ids):
    # HI:  (batch_size, 1)          — graph-level health index
    # out: (batch_size * num_paths, 1) — per-node outputs, ordered by graph then node
    batch_size = HI.shape[0]
    num_paths = out.shape[0] // batch_size          # 28 for a 28-node graph
    path_out = out.reshape(batch_size, num_paths)   # (batch_size, 28)

    return monotonicity_loss(path_out, state_ids, panel_ids)

_DAMAGE_MAP_GEOMETRY_CACHE = {}

def _damage_map_geometry(n_pixels, beta_constant=BETA_CONSTANT):
   
    key = (n_pixels, beta_constant)
    if key in _DAMAGE_MAP_GEOMETRY_CACHE: # Check if the weights have already been computed for this n_pixels and beta_constant
        return _DAMAGE_MAP_GEOMETRY_CACHE[key]

    dA = (PANEL_W * PANEL_H) / n_pixels
    dx = np.sqrt(dA)
    x = np.arange(0, PANEL_W + dx, dx)
    y = np.arange(0, PANEL_H + dx, dx)
    grid = np.meshgrid(x, y, indexing='ij')
    X, Y = grid

    U_arr = np.zeros_like(X)
    _wcpdi_U(U_arr, grid)
    peak_U = U_arr.max()

    W_maps = []
    for a, b in SENSOR_PAIRS:
        p1, p2 = SENSOR_POSITIONS[a - 1], SENSOR_POSITIONS[b - 1]
        d1 = np.sqrt((X - p1[0]) ** 2 + (Y - p1[1]) ** 2)
        d2 = np.sqrt((X - p2[0]) ** 2 + (Y - p2[1]) ** 2)
        d = np.sqrt(((p1 - p2) ** 2).sum())
        path_beta = optimal_beta(d, MAX_DIST,beta_constant)
        R = (d1 + d2) / d - 1
        W_maps.append(np.where(R < path_beta, 1 - R / path_beta, 0.0))
    W_stack = np.stack(W_maps, axis=0)  # (num_paths, nx, ny)

    result = (dx, U_arr, peak_U, W_stack)
    _DAMAGE_MAP_GEOMETRY_CACHE[key] = result
    return result

def damage_map_loss(out, panel_ids, n_pixels=1000, beta=BETA_CONSTANT):
    out = out.reshape(-1)
    num_paths = len(SENSOR_PAIRS)
    batch_size = out.shape[0] // num_paths
    path_out = out.reshape(batch_size, num_paths)

    
    dx, U_arr, peak_U, W_stack = _damage_map_geometry(n_pixels,beta_constant=beta)
    W_stack = torch.tensor(W_stack, dtype=out.dtype, device=out.device)
    U_map = torch.tensor(U_arr, dtype=out.dtype, device=out.device)

    valid_mask = U_map > 0 # Check which pixels are under the influence of at least one sensor path
    U_safe = torch.where(valid_mask, U_map, torch.ones_like(U_map))

    loss = out.sum() * 0.0
    for i in range(batch_size):
        panel_number = int(panel_ids[i].item())

        # multiply sub-HI by its path's W map, then sum over all paths to get the panel's P map
        P_map = (path_out[i].view(-1, 1, 1) * W_stack).sum(dim=0) 
        WCPDI_map = P_map  / (U_safe / peak_U)

        mean_WCPDI = WCPDI_map[valid_mask].mean()

        damage_point = DAMAGE_POINTS[panel_number]
        if damage_point.ndim == 2:
            damage_point = damage_point.mean(axis=0)
        ix = int(np.clip(round(damage_point[0] / dx), 0, WCPDI_map.shape[0] - 1))
        iy = int(np.clip(round(damage_point[1] / dx), 0, WCPDI_map.shape[1] - 1))
        damage_WCPDI = WCPDI_map[ix, iy]

        diff = damage_WCPDI - mean_WCPDI
        loss = loss + ((diff + 10.0) ** 2 - 100.0)

    return loss


def plot_training_history(hist_dict, nr_epochs,save_dir=None):

    epochs = range(1, nr_epochs + 1)
    loss_history =  hist_dict['loss_history']
    val_loss_history = hist_dict['val_loss_history']
    global_loss_history = hist_dict['global_loss_history']
    global_val_loss_history = hist_dict['global_val_loss_history']
    path_loss_history = hist_dict['path_loss_history']
    path_val_loss_history = hist_dict['path_val_loss_history']
    damage_loss_history = hist_dict.get('damage_loss_history')
    damage_val_loss_history = hist_dict.get('damage_val_loss_history')

    # mark the lowest validation loss epoch
    lowest_val_epoch = torch.argmin(val_loss_history[:nr_epochs]).item() + 1

    has_damage_loss = damage_loss_history is not None
    fig, ax = plt.subplots(1, 4 if has_damage_loss else 3, figsize=(24 if has_damage_loss else 18, 5))
    ax[0].plot(epochs, loss_history[:nr_epochs], label='Training Loss')
    ax[0].plot(epochs, val_loss_history[:nr_epochs], label='Validation Loss')
    ax[0].axvline(x=lowest_val_epoch, color='r', linestyle='--', label=f'Lowest Val Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Total Loss')
    ax[0].set_title('Total Loss')
    ax[0].legend()

    ax[1].plot(epochs, global_loss_history[:nr_epochs], label='Training Global Loss')
    ax[1].plot(epochs, global_val_loss_history[:nr_epochs], label='Validation Global Loss')
    ax[1].axvline(x=lowest_val_epoch, color='r', linestyle='--', label=f'Lowest Val Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Global Loss')
    ax[1].set_title('Global Loss')
    ax[1].legend()

    ax[2].plot(epochs, path_loss_history[:nr_epochs], label='Training Path Loss')
    ax[2].plot(epochs, path_val_loss_history[:nr_epochs], label='Validation Path Loss')
    ax[2].axvline(x=lowest_val_epoch, color='r', linestyle='--', label=f'Lowest Val Loss')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Path Loss')
    ax[2].set_title('Path Loss')
    ax[2].legend()

    if has_damage_loss:
        ax[3].plot(epochs, damage_loss_history[:nr_epochs], label='Training Damage Map Loss')
        ax[3].plot(epochs, damage_val_loss_history[:nr_epochs], label='Validation Damage Map Loss')
        ax[3].axvline(x=lowest_val_epoch, color='r', linestyle='--', label=f'Lowest Val Loss')
        ax[3].set_xlabel('Epoch')
        ax[3].set_ylabel('Damage Map Loss')
        ax[3].set_title('Damage Map Loss')
        ax[3].legend()

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir)
    else:
        plt.show()
    plt.close(fig)



def train_with_features(f=DEFAULT_FREQ_INDEX,val_panel=VAL_PANELS,n_epochs=EPOCHS_PER_FOLD_GCN,batch_size=DEFAULT_BATCH_SIZE,lr=LR,out_folder=str(GCN_MODELS_DIR), net_name="gcn_features.pt", cross_validation=False, seed=42):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    num_node_features = DEFAULT_N_FEATURES
    model = DeepGraphCNN(num_node_features=num_node_features).to(device)

    dataset_103 = features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=103, freq=f)
    dataset_104 = features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=104, freq=f)
    dataset_105 = features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=105, freq=f)
    dataset_109 = features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=109, freq=f)

    dict_datasets = {"103": dataset_103, "104": dataset_104, "105": dataset_105, "109": dataset_109}
    val_dataset = dict_datasets[str(val_panel)]

    # compute normalisation stats from training panels only
    all_train_datasets = [dataset for key, dataset in dict_datasets.items() if key != str(val_panel)]
    all_features = torch.cat([data.x for dataset in all_train_datasets for data in dataset], dim=0)
    feature_mean = all_features.mean(dim=0)
    feature_std = all_features.std(dim=0)
    feature_std[feature_std < 1e-10] = 1.0  # avoid division by zero for constant features

    # apply as a transform so it runs on every __getitem__ call (works with InMemoryDataset)
    def norm_transform(data):
        data.x = (data.x - feature_mean) / feature_std
        return data

    for dataset in dict_datasets.values():
        dataset.transform = norm_transform

    train_dataset = torch.utils.data.ConcatDataset(all_train_datasets)
    loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch_geometric.loader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = torch.zeros(n_epochs)
    global_loss_history = torch.zeros(n_epochs)
    path_loss_history = torch.zeros(n_epochs)
    damage_loss_history = torch.zeros(n_epochs)
    val_loss_history = torch.zeros(n_epochs)
    global_val_loss_history = torch.zeros(n_epochs)
    path_val_loss_history = torch.zeros(n_epochs)
    damage_val_loss_history = torch.zeros(n_epochs)

    epochs_done = 0

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        total_global_loss = 0
        total_path_loss = 0
        total_damage_loss = 0
        for data in loader:
            data = data.to(device)

            optimizer.zero_grad()
            out, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)

            global_loss = monotonicity_loss(HI, data.y, data.panel)
            path_loss = combined_path_loss(HI, out, data.y, data.panel)
            damage_loss = damage_map_loss(out, data.panel)
            loss = global_loss + path_loss #+ damage_loss


            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_global_loss += global_loss.item()
            total_path_loss += path_loss.item()
            total_damage_loss += damage_loss.item()
        avg_loss = total_loss / len(loader)
        avg_global_loss = total_global_loss / len(loader)
        avg_path_loss = total_path_loss / len(loader)
        avg_damage_loss = total_damage_loss / len(loader)

        # Validation
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            total_global_val_loss = 0
            total_path_val_loss = 0
            total_damage_val_loss = 0
            for val_data in val_loader:
                val_data = val_data.to(device)
                out_val, HI_val = model(val_data.x.to(device), val_data.edge_index.to(device), val_data.batch.to(device), val_data.edge_weight.to(device))


                global_val_loss = monotonicity_loss(HI_val, val_data.y, val_data.panel)
                path_val_loss = combined_path_loss(HI_val, out_val, val_data.y, val_data.panel)
                damage_val_loss = damage_map_loss(out_val,  val_data.panel)
                loss_val = global_val_loss + path_val_loss #+ damage_val_loss

                total_val_loss += loss_val.item()
                total_global_val_loss += global_val_loss.item()
                total_path_val_loss += path_val_loss.item()
                total_damage_val_loss += damage_val_loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            avg_global_val_loss = total_global_val_loss / len(val_loader)
            avg_path_val_loss = total_path_val_loss / len(val_loader)
            avg_damage_val_loss = total_damage_val_loss / len(val_loader)

        #Save the model with the best validation loss
        if epoch == 0:
            best_val_loss = avg_val_loss
            torch.save({'model': model, 'feature_mean': feature_mean, 'feature_std': feature_std}, os.path.join(out_folder, net_name))
        else:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({'model': model, 'feature_mean': feature_mean, 'feature_std': feature_std}, os.path.join(out_folder, net_name))


        loss_history[epoch] = avg_loss
        val_loss_history[epoch] = avg_val_loss
        global_loss_history[epoch] = avg_global_loss
        path_loss_history[epoch] = avg_path_loss
        damage_loss_history[epoch] = avg_damage_loss
        global_val_loss_history[epoch] = avg_global_val_loss
        path_val_loss_history[epoch] = avg_path_val_loss
        damage_val_loss_history[epoch] = avg_damage_val_loss


        print(f"Epoch: {epoch}, Loss: {avg_loss}, Val Loss: {avg_val_loss}")
        epochs_done += 1

    history_dict = {
        'loss_history': loss_history,
        'global_loss_history': global_loss_history,
        'path_loss_history': path_loss_history,
        'damage_loss_history': damage_loss_history,
        'val_loss_history': val_loss_history,
        'global_val_loss_history': global_val_loss_history,
        'path_val_loss_history': path_val_loss_history,
        'damage_val_loss_history': damage_val_loss_history}

    if cross_validation:
        plot_output = os.path.join(out_folder, f"training_history_{net_name.split('.')[0]}_cv.png")
    plot_training_history(history_dict, epochs_done, save_dir=plot_output if cross_validation else None)


class EnsembleGCN(torch.nn.Module):
    """Wraps N cross-validation GCN models into one module"""
    def __init__(self, checkpoints):
        # checkpoints: list of {'model', 'feature_mean', 'feature_std'} dicts
        super().__init__()
        self.models = torch.nn.ModuleList([ck['model'] for ck in checkpoints])
        self.register_buffer('feature_means', torch.stack([ck['feature_mean'] for ck in checkpoints]))
        self.register_buffer('feature_stds',  torch.stack([ck['feature_std']  for ck in checkpoints]))

    def forward(self, x, edge_index, batch, edge_weight):
        all_out, all_HI = [], []
        for model, mean, std in zip(self.models, self.feature_means, self.feature_stds):
            x_norm = (x - mean) / std
            out, HI = model(x_norm, edge_index, batch, edge_weight)
            all_out.append(out)
            all_HI.append(HI)
        # out: (n_models, batch*n_paths, 1) → mean over models
        # HI:  (n_models, batch, 1)         → mean over models
        mean_out = torch.stack(all_out, dim=0).mean(dim=0)
        mean_HI  = torch.stack(all_HI,  dim=0).mean(dim=0)
        return mean_out, mean_HI


def build_and_save_ensemble(model_path, save_path):
    ensemble_filename = os.path.basename(save_path)
    checkpoints = [
        torch.load(os.path.join(model_path, f), weights_only=False)
        for f in sorted(os.listdir(model_path))
        if f.endswith('.pt') and f != ensemble_filename
    ]
    ensemble = EnsembleGCN(checkpoints)
    torch.save(ensemble, save_path)
    print(f"Ensemble saved to {save_path} ({len(checkpoints)} models)")
    return ensemble


def ensemble_predict(datasets, test_dataset, model_path, save_dir=None, datasets_by_fold=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_datasets = datasets + [test_dataset]
    n = len(all_datasets)

    # HI predictions from each model: list of lists (dataset → model predictions)
    all_HI_per_dataset = [[] for _ in all_datasets]
    states_per_dataset = [None] * n

    for model_file in sorted(os.listdir(model_path)):
        if not model_file.endswith('.pt') or model_file == 'ensemble_model.pt':
            continue
        checkpoint = torch.load(os.path.join(model_path, model_file), weights_only=False)
        model = checkpoint['model'].to(device)
        feature_mean = checkpoint['feature_mean']
        feature_std = checkpoint['feature_std']
        model.eval()

        #Once traing the model on CAE latent space, each val. fold is extracting different latent spaces, so only datasets corresponfing to the given fold are taken into account 
        fold = checkpoint.get('fold')
        if datasets_by_fold is not None and fold in datasets_by_fold:
            model_datasets = datasets_by_fold[fold]
        else:
            model_datasets = all_datasets

        def norm_transform(data, mean=feature_mean, std=feature_std):
            data.x = (data.x - mean) / std
            return data

        for i, dataset in enumerate(model_datasets):
            dataset.transform = norm_transform
            loader = torch_geometric.loader.DataLoader(dataset, batch_size=15, shuffle=False)
            batch_HI, batch_states = [], []
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    _, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)
                    batch_HI.append(HI.cpu())
                    batch_states.append(data.y.cpu())
            all_HI_per_dataset[i].append(torch.cat(batch_HI).squeeze().numpy())
            if states_per_dataset[i] is None:
                states_per_dataset[i] = torch.cat(batch_states).squeeze().numpy()

    fig, ax = plt.subplots(1, n, figsize=(5 * n, 5))
    colors = ['steelblue'] * len(datasets) + ['orange']
    for i, dataset in enumerate(all_datasets):
        states = states_per_dataset[i]
        stacked = np.stack(all_HI_per_dataset[i], axis=0)  # (n_models, n_states)
        mean_HI = stacked.mean(axis=0)
        std_HI = stacked.std(axis=0)

        order = np.argsort(states)
        ax[i].fill_between(states[order], (mean_HI - std_HI)[order], (mean_HI + std_HI)[order],
                           alpha=0.25, color=colors[i])
        ax[i].scatter(states, mean_HI, alpha=0.7, s=15, color=colors[i])
        label = 'Test' if dataset is test_dataset else 'Train'
        ax[i].set_title(f'Panel {dataset.panel_number} ({label})')
        ax[i].grid(True)

    fig.suptitle('Ensemble Health Index vs Damage State')
    fig.supxlabel('Damage State Index')
    fig.supylabel('Health Index (HI)')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir)
    else:
        plt.show()
    plt.close(fig)

def plot_HI(model, train_datasets, validation_datasets, test_datasets, save_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    total_len_datasets = len(train_datasets) + len(validation_datasets) + len(test_datasets)
    fig, ax = plt.subplots(1, total_len_datasets, figsize=(5 * total_len_datasets, 5))

    plot_idx = 0
    for dataset in train_datasets:
        loader = torch_geometric.loader.DataLoader(dataset, batch_size=15, shuffle=False)
        all_HI = []
        all_states = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                _, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)
                all_HI.append(HI.cpu())
                all_states.append(data.y.cpu())
        all_HI = torch.cat(all_HI).squeeze().numpy()
        all_states = torch.cat(all_states).squeeze().numpy()

        ax[plot_idx].scatter(all_states, all_HI, alpha=0.7)
        ax[plot_idx].set_title(f'Panel {dataset.panel_number}')
        ax[plot_idx].grid(True)
        plot_idx += 1

    for dataset in validation_datasets:
        loader = torch_geometric.loader.DataLoader(dataset, batch_size=15, shuffle=False)
        all_HI = []
        all_states = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                _, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)
                all_HI.append(HI.cpu())
                all_states.append(data.y.cpu())
        all_HI = torch.cat(all_HI).squeeze().numpy()
        all_states = torch.cat(all_states).squeeze().numpy()

        ax[plot_idx].scatter(all_states, all_HI, alpha=0.7, color='orange')
        ax[plot_idx].set_title(f'Panel {dataset.panel_number} (Validation)')
        ax[plot_idx].grid(True)
        plot_idx += 1

       
    for dataset in test_datasets:
        loader = torch_geometric.loader.DataLoader(dataset, batch_size=15, shuffle=False)
        all_HI = []
        all_states = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                _, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)
                all_HI.append(HI.cpu())
                all_states.append(data.y.cpu())
        all_HI = torch.cat(all_HI).squeeze().numpy()
        all_states = torch.cat(all_states).squeeze().numpy()

        ax[plot_idx].scatter(all_states, all_HI, alpha=0.7, color='green')
        ax[plot_idx].set_title(f'Panel {dataset.panel_number} (Test)')
        ax[plot_idx].grid(True)
        plot_idx += 1
    fig.suptitle('Learned Health Index vs Damage State')
    fig.supxlabel('Damage State Index')
    fig.supylabel('Health Index (HI)')
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir)
    else:
        plt.show()
    plt.close(fig)

def plot_sHI(model, dataset, path_idx):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=15, shuffle=False)
    all_sHI = []
    all_states = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _ = model(data.x, data.edge_index, data.batch, data.edge_weight)
            # out shape: (batch_size * num_paths, 1)
            num_paths = 28
            path_out = out.reshape(-1, num_paths)  # shape: (batch_size, num_paths)
            sHI_for_path = path_out[:, path_idx]    # shape: (batch_size,)
            all_sHI.append(sHI_for_path.cpu())
            all_states.append(data.y.cpu())
    all_sHI = torch.cat(all_sHI).squeeze().numpy()
    all_states = torch.cat(all_states).squeeze().numpy()
    plt.figure(figsize=(10, 5))
    plt.scatter(all_states, all_sHI, alpha=0.7)
    plt.xlabel('Damage State Index')
    plt.ylabel(f'sHI for Path {path_idx + 1}')
    plt.title(f'Learned sHI vs Damage State Panel {dataset.panel_number} Path {path_idx + 1}')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

  
    validation_panels = [int(p) for p in BASE_PANELS]
    test_panel = int(TEST_PANEL[0])

    dataset_dict = {
        int(p): features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=int(p), freq=DEFAULT_FREQ_INDEX)
        for p in BASE_PANELS + TEST_PANEL
    }
    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = str(CROSS_VALIDATION_RESULTS_DIR / current_date)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for val_panel in validation_panels:
        
        model_name = "gcn_features_val_" + str(val_panel) + ".pt"
        train_with_features(batch_size=30, val_panel=val_panel, net_name=model_name, n_epochs=800, lr=0.001, out_folder=output_dir, cross_validation=True, seed=42)

        bundle = torch.load(f"{output_dir}/{model_name}", weights_only=False)
       
        if isinstance(bundle, dict):
            model = bundle['model']
            feature_mean = bundle['feature_mean']
            feature_std  = bundle['feature_std']
            def norm_transform(data):
                data.x = (data.x - feature_mean) / feature_std
                return data
            for ds in dataset_dict.values():
                ds.transform = norm_transform
        else:
            model = bundle

        train_datasets = [dataset for key, dataset in dataset_dict.items() if key != val_panel and key != test_panel]
        validation_datasets = [dataset_dict[val_panel]]
        test_datasets = [dataset_dict[test_panel]]
        plot_HI(model, train_datasets, validation_datasets, test_datasets, save_dir=f"{output_dir}/HI_plot_val_{val_panel}.png")

    ensemble_predict([dataset_dict[p] for p in validation_panels], dataset_dict[test_panel], model_path=output_dir, save_dir=f"{output_dir}/ensemble_HI_plot.png")

    # Build and save ensemble model
    ensemble_model_path = os.path.join(output_dir, "ensemble_model.pt")
    build_and_save_ensemble(output_dir, ensemble_model_path)

    # plot sHI for an example path
    example_path_idx = 0  # Change this index to plot sHI for a different path
    plot_sHI(model, dataset_dict[test_panel], example_path_idx)

    