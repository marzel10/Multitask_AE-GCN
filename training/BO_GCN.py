"""
Bayesian hyperparameter optimization for DeepGraphCNN (models/GCN.py), trained via a
leave-one-out cross-validation loop over BASE_PANELS

Hyperparameters tuned:
    - batch_size, learning_rate
    - nr_hidden_channels (0 or 1): whether DeepGraphCNN has one hidden GCNConv layer
      between the fixed 24-dim input compression and the final scalar output layer
    - hidden_dim: width of that hidden layer (only used when nr_hidden_channels == 1,
      but always registered as an hp -- see get_hidden_dim_hp)
    - dropout: dropout probability for the hidden layer 

Fixed (not tuned):
    - num_node_features: always DEFAULT_GCN_FEATURES (the AE's latent_dim -- every
      Panel_GraphDataset here is built with big_latent=True), compressed to a fixed
      INPUT_COMPRESS_DIM=24 via DeepGraphCNN's input_compress_dim (mirrors GraphCNN's
      own Linear(num_node_features, 24) step)
    - use_residual: always True

Search objective: "Objective" (direction="max") =
    mean_fitness - DAMAGE_LOSS_WEIGHT * mean_damage_loss
where mean_fitness is a*monotonicity + b*trendability + c*prognosability

After the search: retrains one model per CV_PANELS fold with the best hyperparameters
(saving each as a GCN_train.py-compatible checkpoint dict), builds + saves an
EnsembleGCN from those folds, and reproduces the same diagnostic plots as
BO_features.py (progress, running best, overfitting, HP sensitivity, HP coverage) plus
GCN_train.py's own training-history / HI plots per fold and for the ensemble.
"""
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import json

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "tools", "training", "intermediate_results_check", "results_analysis"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os

import numpy as np
import pandas as pd
import keras_tuner as kt
import matplotlib.pyplot as plt
import torch
import torch_geometric

from GCN import DeepGraphCNN
from graph_dataset import Panel_GraphDataset, features_GraphDataset
from GCN_train import (
    monotonicity_loss, combined_path_loss, damage_map_loss,
    plot_training_history, plot_HI, build_and_save_ensemble, ensemble_predict,
)
from prognostic_criteria import monotonicity_criterion, trendability_criterion, prognosability_criterion
from config import (
    BETA_CONSTANT, ENABLE_DAMAGE_LOSS,ENABLE_PATH_LOSS, ENABLE_GLOBAL_LOSS, GRAPH_DATA_DIR, BO_TUNER_DIR, BO_SEARCH_RESULTS_DIR,
    DEFAULT_FREQ_INDEX, DEFAULT_GCN_FEATURES, DEFAULT_N_FEATURES, EPOCHS_PER_FOLD_GCN, CV_PANELS, MAX_TRIALS_GCN, TEST_PANEL, DAMAGE_LOSS_WEIGHT, TEST_RUN_DIR, VAL_PANELS,
)

CV_PANELS_INT = [int(p) for p in CV_PANELS] 
TEST_PANEL_INT = int(TEST_PANEL[0]) 

TYPE = "peak"  # default adjacency matrix type
RAW_FEATURES = False  # default to using AE latent features, not raw features
HP_COLS = ["batch_size", "learning_rate", "nr_hidden_channels", "hidden_dim", "dropout"]


def _input_dims(raw_features, big_latent):
    if not big_latent:
        return 1, 1  # sHI mode: single input feature
    n = DEFAULT_N_FEATURES if raw_features else DEFAULT_GCN_FEATURES    
    return n, n


# ─── HP helpers ────────────────────────────────────────────────────────────────
def get_batch_size_hp(hp):
    return hp.Int("batch_size", min_value=14, max_value=32, step=4)


def get_lr_hp(hp):
    return hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")


def get_nr_hidden_channels_hp(hp, raw_features):
    if raw_features:
        return hp.Fixed("nr_hidden_channels", 1)
    return hp.Int("nr_hidden_channels", min_value=0, max_value=1)


def get_hidden_dim_hp(hp, raw_features, big_latent=True):
    # Unused in build_model when nr_hidden_channels == 0.
    if raw_features: # make bigger model for raw features
        return hp.Int("hidden_dim", min_value=32, max_value=128, step=8)
    if not big_latent:  # sHI mode: single input feature, no compression, smaller hidden layer
        return hp.Int("hidden_dim", min_value=1, max_value=10, step=1)
    return hp.Int("hidden_dim", min_value=8, max_value=64, step=8)

def get_dropout_hp(hp):
    return hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)


# ─── Model builder ──────────────────────────────────────────────────────────────
def build_model(hp, raw_features=RAW_FEATURES, big_latent=True):
    nr_hidden = get_nr_hidden_channels_hp(hp, raw_features)
    hidden_dim = get_hidden_dim_hp(hp, raw_features, big_latent)
    dropout = get_dropout_hp(hp)

    num_node_features, input_compress_dim = _input_dims(raw_features, big_latent)
    hidden_channels = (hidden_dim,) if nr_hidden == 1 else ()
    
    return DeepGraphCNN(
        num_node_features=num_node_features,
        hidden_channels=hidden_channels,
        dropout=dropout,
        use_residual=True if nr_hidden == 1 else False,
        input_compress_dim=input_compress_dim,
    )


def _build_from_params(params, raw_features=RAW_FEATURES, big_latent=True):
    """Same as build_model, but from a plain {hp_name: value}, used by the post-search retrain loop."""
    num_node_features, input_compress_dim = _input_dims(raw_features, big_latent)
    hidden_channels = (params["hidden_dim"],) if params["nr_hidden_channels"] == 1 else ()
    return DeepGraphCNN(
        num_node_features=num_node_features,
        hidden_channels=hidden_channels,
        dropout=params["dropout"],
        use_residual=True if params["nr_hidden_channels"] == 1 else False,
        input_compress_dim=input_compress_dim,
    )


# ─── Fitness (prognostic_criteria.py) ────────────────────────────────────────────
def _collect_HI(model, dataset, device):
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=32, shuffle=False)
    all_HI, all_states = [], []
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            _, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)
            all_HI.append(HI.cpu().numpy().reshape(-1))
            all_states.append(data.y.cpu().numpy().reshape(-1))
    HI = np.concatenate(all_HI)
    states = np.concatenate(all_states)
    return HI[np.argsort(states)]


def fitness_objective(model, datasets, device, a=1.0, b=1.0, c=1.0):
    DI = [_collect_HI(model, ds, device) for ds in datasets]
    monotonicity = monotonicity_criterion(DI)
    trendability = trendability_criterion(DI)
    prognosability = prognosability_criterion(DI)
    return a * monotonicity + b * trendability + c * prognosability


# ─── Custom tuner ─────────────────────────────────────────────────────────────
class MyGCNTuner(kt.BayesianOptimization):
    def __init__(self, *args, freq=DEFAULT_FREQ_INDEX, cv_panels=None,
                 epochs_per_fold=EPOCHS_PER_FOLD_GCN, type=TYPE, raw_features=RAW_FEATURES, beta=BETA_CONSTANT, damage_loss=ENABLE_DAMAGE_LOSS, path_loss=ENABLE_PATH_LOSS, global_loss=ENABLE_GLOBAL_LOSS, seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.cv_panels = cv_panels or CV_PANELS_INT
        self.BO_val_panels = VAL_PANELS if VAL_PANELS else None
        self.epochs_per_fold = epochs_per_fold
        self.type = type
        self.big_latent = True if (type != "sHI_by_area" and type != "sHI_peak") else False
        self.raw_features = raw_features
        self.beta = beta
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.damage_loss = damage_loss
        self.path_loss = path_loss
        self.global_loss = global_loss
        self.seed = seed

    def _log_damage_trajectory(self, trial_id, val_panel, train_hist, val_hist):
        log_dir = os.path.join(str(BO_TUNER_DIR), "trial_damage_logs")
        os.makedirs(log_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(train_hist, label="Training Damage Map Loss (raw)")
        ax.plot(val_hist, label="Validation Damage Map Loss (raw)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Damage Map Loss")
        ax.set_title(f"Trial {trial_id} — val panel {val_panel}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(log_dir, f"trial_{trial_id}_val_{val_panel}.svg"))
        plt.close(fig)

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        bs = get_batch_size_hp(hp)
        lr = get_lr_hp(hp)

        fold_fitness, fold_damage, fold_val_loss, fold_train_loss = [], [], [], []

        # leave-one-out over cv_panels
        for val_panel in self.BO_val_panels if self.BO_val_panels is not None else self.cv_panels:
            val_panel = int(val_panel)

            if self.raw_features:
                datasets = {
                    panel: features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=panel, freq=self.freq, type=self.type, beta_constant=self.beta)
                    for panel in self.cv_panels
                }
            else:
                datasets = {
                    panel: Panel_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=panel, freq=self.freq, big_latent=self.big_latent, type=self.type, fold=val_panel, beta_constant=self.beta)
                    for panel in self.cv_panels
                }

            train_panels = [p for p in self.cv_panels if p != val_panel]

            # normalization stats from training panels only (mirrors _retrain_fold)
            all_train_datasets = [datasets[p] for p in train_panels]
            all_features = torch.cat([data.x for ds in all_train_datasets for data in ds], dim=0)
            feature_mean = all_features.mean(dim=0)
            feature_std = all_features.std(dim=0)
            feature_std[feature_std < 1e-10] = 1.0

            def norm_transform(data, mean=feature_mean, std=feature_std):
                data.x = (data.x - mean) / std
                return data

            for ds in datasets.values():
                ds.transform = norm_transform

            train_ds = torch.utils.data.ConcatDataset([datasets[p] for p in train_panels])
            train_loader = torch_geometric.loader.DataLoader(train_ds, batch_size=bs, shuffle=True)
            val_loader = torch_geometric.loader.DataLoader(datasets[val_panel], batch_size=bs, shuffle=False)

            torch.manual_seed(self.seed)
            model = self.hypermodel.build(hp).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            best_val_loss = float("inf")
            best_train_loss_at_best = float("inf")
            best_state = None
            patience, patience_counter = 10, 0
            damage_loss_history, damage_val_loss_history = [], []

            for epoch in range(self.epochs_per_fold):
                model.train()
                total_train_loss = 0.0
                total_train_damage = 0.0
                n_train_batches = 0
                for data in train_loader:
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    out, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)

                    Hi_loss = monotonicity_loss(HI, data.y, data.panel) if self.global_loss else 0.0
                    path = combined_path_loss(HI, out, data.y, data.panel) if self.path_loss else 0.0
                    map_loss = damage_map_loss(out, data.panel, beta=self.beta) if self.damage_loss else 0.0
                    loss = Hi_loss + path + DAMAGE_LOSS_WEIGHT * map_loss
                    if self.damage_loss:
                        total_train_damage += map_loss.item()

                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()
                    n_train_batches += 1
                avg_train_loss = total_train_loss / max(n_train_batches, 1)
                damage_loss_history.append(total_train_damage / max(n_train_batches, 1))

                model.eval()
                total_val_loss = 0.0
                total_val_damage = 0.0
                n_val_batches = 0
                with torch.no_grad():
                    for data in val_loader:
                        data = data.to(self.device)
                        out, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)

                        HI_loss = monotonicity_loss(HI, data.y, data.panel) if self.global_loss else 0.0
                        path = combined_path_loss(HI, out, data.y, data.panel) if self.path_loss else 0.0
                        map_loss = damage_map_loss(out, data.panel, beta=self.beta) if self.damage_loss else 0.0
                        loss = HI_loss + path + DAMAGE_LOSS_WEIGHT * map_loss

                        if self.damage_loss:
                            total_val_damage += map_loss.item()

                        total_val_loss += loss.item()
                        n_val_batches += 1
                avg_val_loss = total_val_loss / max(n_val_batches, 1)
                damage_val_loss_history.append(total_val_damage / max(n_val_batches, 1))

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_train_loss_at_best = avg_train_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            if self.damage_loss:
                self._log_damage_trajectory(trial.trial_id, val_panel, damage_loss_history, damage_val_loss_history)

            if best_state is not None:
                model.load_state_dict(best_state)

            fold_val_loss.append(best_val_loss)
            fold_train_loss.append(best_train_loss_at_best)

            # damage_map_loss on the held-out validation panel, using this fold's model
            model.eval()
            total_damage_loss, n_batches = 0.0, 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(self.device)
                    out, _ = model(data.x, data.edge_index, data.batch, data.edge_weight)
                    total_damage_loss += damage_map_loss(out,  data.panel, beta=self.beta).item()
                    n_batches += 1
            fold_damage.append(total_damage_loss / max(n_batches, 1))

            fold_fitness.append(
                fitness_objective(model, [datasets[p] for p in self.cv_panels], self.device)
            )

            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_fitness = float(np.mean(fold_fitness))
        mean_damage = float(np.mean(fold_damage))
        objective = mean_fitness - DAMAGE_LOSS_WEIGHT * mean_damage

        return {
            "Objective": objective,
            "mean_fitness": mean_fitness,
            "mean_damage_loss": mean_damage,
            "mean_train_loss": float(np.mean(fold_train_loss)),
            "mean_val_loss": float(np.mean(fold_val_loss)),
        }


# ─── Retrain (best hyperparameters, one model per CV fold) ────────────────────
def _retrain_fold(best_params, val_panel, freq, out_dir, epochs, seed=42, type=TYPE, raw_features=RAW_FEATURES,beta=BETA_CONSTANT, add_damage_loss=ENABLE_DAMAGE_LOSS, add_path_loss=ENABLE_PATH_LOSS, add_global_loss=ENABLE_GLOBAL_LOSS):
    """Retrain one leave-one-out fold with the given (best) hyperparameter values."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    big_latent = True if (type != "sHI_by_area" and type != "sHI_peak") else False

    if raw_features:
        datasets = {p: features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=p, freq=freq, type=type, beta_constant=beta) for p in CV_PANELS_INT}
        test_dataset = features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=TEST_PANEL_INT, freq=freq, type=type, beta_constant=beta)
    else:
        datasets = {p: Panel_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=p, freq=freq, big_latent=big_latent, type=type,beta_constant=beta, fold=val_panel) for p in CV_PANELS_INT}
        test_dataset = Panel_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=TEST_PANEL_INT, freq=freq, big_latent=big_latent, type=type,beta_constant=beta, fold=val_panel)
    
    train_panels = [p for p in CV_PANELS_INT if p != val_panel]

    # normalization stats from training panels only (mirrors GCN_train.train_with_features)
    all_train_datasets = [datasets[p] for p in train_panels]
    all_features = torch.cat([data.x for ds in all_train_datasets for data in ds], dim=0)
    feature_mean = all_features.mean(dim=0)
    feature_std = all_features.std(dim=0)
    feature_std[feature_std < 1e-10] = 1.0

    def norm_transform(data):
        data.x = (data.x - feature_mean) / feature_std
        return data

    for ds in list(datasets.values()) + [test_dataset]:
        ds.transform = norm_transform

    model = _build_from_params(best_params, raw_features=raw_features, big_latent=big_latent).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])

    train_ds = torch.utils.data.ConcatDataset(all_train_datasets)
    loader = torch_geometric.loader.DataLoader(train_ds, batch_size=best_params["batch_size"], shuffle=True)
    val_loader = torch_geometric.loader.DataLoader(datasets[val_panel], batch_size=best_params["batch_size"], shuffle=False)

    loss_history = torch.zeros(epochs)
    global_loss_history = torch.zeros(epochs)
    path_loss_history = torch.zeros(epochs)
    damage_loss_history = torch.zeros(epochs)
    val_loss_history = torch.zeros(epochs)
    global_val_loss_history = torch.zeros(epochs)
    path_val_loss_history = torch.zeros(epochs)
    damage_val_loss_history = torch.zeros(epochs)

    net_name = f"gcn_bo_val_{val_panel}.pt"
    ckpt_path = os.path.join(out_dir, net_name)
    best_val_loss = None
    epochs_done = 0

    for epoch in range(epochs):
        model.train()
        total_loss = total_global = total_path = total_damage = 0.0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)
            global_loss = monotonicity_loss(HI, data.y, data.panel) if add_global_loss else 0.0
            path_loss = combined_path_loss(HI, out, data.y, data.panel) if add_path_loss else 0.0
            damage_loss = damage_map_loss(out, data.panel, beta=beta) if add_damage_loss else 0.0
            loss = global_loss + path_loss + DAMAGE_LOSS_WEIGHT * damage_loss 
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if add_global_loss: total_global += global_loss.item()
            if add_path_loss: total_path += path_loss.item()
            if add_damage_loss: total_damage += damage_loss.item()
        n = len(loader)
        avg_loss, avg_global, avg_path, avg_damage = total_loss / n, total_global / n, total_path / n, total_damage / n

        model.eval()
        total_vloss = total_vglobal = total_vpath = total_vdamage = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)
                global_loss = monotonicity_loss(HI, data.y, data.panel) if add_global_loss else 0.0
                path_loss = combined_path_loss(HI, out, data.y, data.panel) if add_path_loss else 0.0
                damage_loss = damage_map_loss(out, data.panel, beta=beta) if add_damage_loss else 0.0

                loss = global_loss + path_loss + DAMAGE_LOSS_WEIGHT * damage_loss

                total_vloss += loss.item()
                if add_global_loss: total_vglobal += global_loss.item()
                if add_path_loss: total_vpath += path_loss.item()
                if add_damage_loss: total_vdamage += damage_loss.item()
        nv = len(val_loader)
        avg_vloss, avg_vglobal = total_vloss / nv, total_vglobal / nv
        avg_vpath, avg_vdamage = total_vpath / nv, total_vdamage / nv

        loss_history[epoch] = avg_loss; global_loss_history[epoch] = avg_global
        path_loss_history[epoch] = avg_path; damage_loss_history[epoch] = avg_damage
        val_loss_history[epoch] = avg_vloss; global_val_loss_history[epoch] = avg_vglobal
        path_val_loss_history[epoch] = avg_vpath; damage_val_loss_history[epoch] = avg_vdamage
        epochs_done += 1

        if best_val_loss is None or avg_vloss < best_val_loss:
            best_val_loss = avg_vloss
            torch.save({'model': model, 'feature_mean': feature_mean, 'feature_std': feature_std, 'fold': val_panel}, ckpt_path)

    history_dict = {
        'loss_history': loss_history, 'val_loss_history': val_loss_history,
        'global_loss_history': global_loss_history, 'global_val_loss_history': global_val_loss_history,
        'path_loss_history': path_loss_history, 'path_val_loss_history': path_val_loss_history,
        'damage_loss_history': damage_loss_history, 'damage_val_loss_history': damage_val_loss_history,
    }
    plot_training_history(history_dict, epochs_done,
                           save_dir=os.path.join(out_dir, f"learning_curves_val_{val_panel}.svg"))

    checkpoint = torch.load(ckpt_path, weights_only=False)
    fold_model = checkpoint['model']
    plot_HI(
        fold_model,
        train_datasets=[datasets[p] for p in train_panels],
        validation_datasets=[datasets[val_panel]],
        test_datasets=[test_dataset],
        save_dir=os.path.join(out_dir, f"HI_plot_val_{val_panel}.svg"),
    )
    return net_name


# ─── Helpers for parsing tuner results ────────────────────────────────────────
def _scalar(v):
    if hasattr(v, "value"):
        v = v.value
    if isinstance(v, (list, tuple)):
        return float(v[0])
    return float(v)


def _collect_metric_histories(tuner):
    fitness_h, damage_h, val_h, train_h = [], [], [], []
    for t in tuner.oracle.trials.values():
        if t.score is None:
            continue
        fitness_obs = [_scalar(o.value) for o in t.metrics.get_history("mean_fitness")]
        damage_obs = [_scalar(o.value) for o in t.metrics.get_history("mean_damage_loss")]
        val_obs = [_scalar(o.value) for o in t.metrics.get_history("mean_val_loss")]
        train_obs = [_scalar(o.value) for o in t.metrics.get_history("mean_train_loss")]
        if fitness_obs and damage_obs and val_obs and train_obs:
            fitness_h.append(max(fitness_obs))
            damage_h.append(min(damage_obs))
            val_h.append(min(val_obs))
            train_h.append(min(train_obs))
    return fitness_h, damage_h, train_h, val_h


def _build_trials_dataframe(tuner):
    records = []
    for tid, t in tuner.oracle.trials.items():
        if t.score is None:
            continue
        row = {"trial_id": tid, "objective": t.score}
        row.update(t.hyperparameters.values)

        fitness_hist = [_scalar(o) for o in t.metrics.get_history("mean_fitness")]
        damage_hist = [_scalar(o) for o in t.metrics.get_history("mean_damage_loss")]
        val_hist = [_scalar(o) for o in t.metrics.get_history("mean_val_loss")]
        train_hist = [_scalar(o) for o in t.metrics.get_history("mean_train_loss")]

        row["final_fitness"] = fitness_hist[-1] if fitness_hist else None
        row["final_damage_loss"] = damage_hist[-1] if damage_hist else None
        row["final_val_loss"] = val_hist[-1] if val_hist else None
        row["final_train_loss"] = train_hist[-1] if train_hist else None
        row["overfit_gap"] = (
            row["final_val_loss"] - row["final_train_loss"]
            if (row["final_val_loss"] is not None and row["final_train_loss"] is not None)
            else None
        )
        records.append(row)
    return pd.DataFrame(records).sort_values("objective", ascending=False)


# ─── Plotting ────────────────────────────
def _min_max_normalize(values):
    x = np.asarray(values, dtype=float)
    lo, hi = x.min(), x.max()
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def plot_optimization_progress(fitness_h, damage_h, save_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(_min_max_normalize(fitness_h), label="Fitness (min-max)")
    ax.plot(_min_max_normalize(damage_h), label="Damage map loss (min-max)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Min-Max Normalized [0, 1]")
    ax.set_title("Fitness & Damage Map Loss per Trial")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_running_best(objective_h, save_path):
    running_best = np.maximum.accumulate(objective_h) if len(objective_h) else np.array([])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(running_best, "k--", label="Running Best")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best Objective")
    ax.set_title("Running Best — GCN BO")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return running_best


def plot_overfitting(df, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(df["final_train_loss"], df["final_val_loss"],
                     c=df["overfit_gap"], cmap="RdYlGn_r", edgecolors="k", s=60)
    fig.colorbar(sc, ax=ax, label="overfit gap (val - train)")
    lo, hi = df["final_train_loss"].min(), df["final_train_loss"].max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="no overfit line")
    ax.set_xlabel("Final train loss")
    ax.set_ylabel("Final val loss")
    ax.set_title("Overfitting — GCN BO")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_hp_sensitivity(df, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), squeeze=False)
    for ax, col in zip(axes.flat, HP_COLS):
        ax.scatter(df[col], df["objective"], alpha=0.6, edgecolors="k", linewidths=0.3)
        ax.set_xlabel(col); ax.set_ylabel("objective")
        ax.set_title(f"{col} vs objective")
    for ax in axes.flat[len(HP_COLS):]:
        ax.axis("off")
    fig.suptitle("HP Sensitivity — GCN BO", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_hp_coverage(df, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), squeeze=False)
    for ax, col in zip(axes.flat, HP_COLS):
        ax.hist(df[col].dropna(), bins=10, edgecolor="k")
        ax.set_title(col)
    for ax in axes.flat[len(HP_COLS):]:
        ax.axis("off")
    fig.suptitle("Sampled HP Distribution — GCN BO", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ─── Main optimization routine ────────────────────────────────────────────────
def run_bayesian_optimization(freq=DEFAULT_FREQ_INDEX, max_trials=MAX_TRIALS_GCN, out_dir=None,
                               epochs_per_fold=EPOCHS_PER_FOLD_GCN, retrain_epochs=None, type=TYPE, raw_features=RAW_FEATURES, beta=BETA_CONSTANT):
    t_start = pd.Timestamp.now()
    beta_suffix = f"_beta{beta}" if beta != BETA_CONSTANT else "" # for sensitivity sweeps on beta
    project_name = f"gcn_{type}{'_raw' if raw_features else ''}{beta_suffix}"
    if type == "peak_tff":
        damage_loss = True
        path_loss = False
        global_loss = False
    elif type == "peak_fft":
        damage_loss = False
        path_loss = False
        global_loss = True
    elif type == "peak_tft":
        damage_loss = True
        path_loss = False
        global_loss = True
    else:
        damage_loss = ENABLE_DAMAGE_LOSS
        path_loss = ENABLE_PATH_LOSS
        global_loss = ENABLE_GLOBAL_LOSS
    big_latent = True 

    if type=="peak" and raw_features:
        do_BO = True
    else:
        do_BO = False

   
    if raw_features and type=="peak":
        out_dir = str(BO_SEARCH_RESULTS_DIR / f"Bayesian_GCN_raw_freq{freq}{beta_suffix}") if out_dir is None else out_dir
    else:
        out_dir = str(BO_SEARCH_RESULTS_DIR / f"Bayesian_GCN_{type}_freq{freq}{beta_suffix}") if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)

    if do_BO:
        tuner = MyGCNTuner(
            partial(build_model, raw_features=raw_features, big_latent=big_latent),
            objective=kt.Objective("Objective", direction="max"),
            max_trials=max_trials,
            directory=str(BO_TUNER_DIR),
            project_name=project_name,
            overwrite=True,
            freq=freq,
            epochs_per_fold=epochs_per_fold,
            type=type,
            raw_features=raw_features,
            beta=beta,
            damage_loss=damage_loss,
            path_loss=path_loss,
            global_loss=global_loss,
        )
        tuner.search()
        t_elapsed = pd.Timestamp.now() - t_start
        print(f"GCN Bayesian optimization completed in {t_elapsed}")


        best = tuner.oracle.get_best_trials(1)[0]
        print("Trial ID:", best.trial_id)
        print("Objective (mean_fitness - damage_weight*mean_damage_loss):", best.score)
        print("Hyperparameters:", best.hyperparameters.values)
        print("Final mean_fitness:", best.metrics.get_last_value("mean_fitness"))
        print("Final mean_damage_loss:", best.metrics.get_last_value("mean_damage_loss"))

        with open(f"{out_dir}/best_trial_details.txt", "w") as f:
            f.write(f"Optimization time {t_elapsed}\n")
            f.write(f"Trial ID: {best.trial_id}\n")
            f.write(f"Objective: {best.score}\n")
            f.write(f"Hyperparameters: {best.hyperparameters.values}\n")
            f.write(f"Final mean_fitness: {best.metrics.get_last_value('mean_fitness')}\n")
            f.write(f"Final mean_damage_loss: {best.metrics.get_last_value('mean_damage_loss')}\n")
            f.write(f"Final mean_train_loss: {best.metrics.get_last_value('mean_train_loss')}\n")
            f.write(f"Final mean_val_loss: {best.metrics.get_last_value('mean_val_loss')}\n")

    # save hyperameters in json format for later use
    if not raw_features and type=="peak":
        params_path = TEST_RUN_DIR/"results"/f"best_hyperparameters_freq{freq}{beta_suffix}_pre_processed.json"
    else:
        params_path = TEST_RUN_DIR/"results"/f"best_hyperparameters_freq{freq}{beta_suffix}.json"
    if do_BO:
        best_params = best.hyperparameters.values
        # save the best hyperparameters as a dictionary for other paths
        params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"Saved: {params_path}")
    else:
        # Load the best hyperparameters found for another path at this freq/model_type
        with open(params_path, "r") as f:
            best_params = json.load(f)

    # ── Retrain: one model per CV fold with the best hyperparameters ──────────
    
    fold_net_names = []
    for val_panel in CV_PANELS_INT:
        net_name = _retrain_fold(best_params, val_panel, freq, out_dir,
                                  epochs=retrain_epochs or epochs_per_fold,
                                  type=type, raw_features=raw_features, beta=beta, add_damage_loss=damage_loss, add_path_loss=path_loss, add_global_loss=global_loss)
        fold_net_names.append(net_name)


    if raw_features:
        cv_datasets = [features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=p, freq=freq, type=type, beta_constant=beta) for p in CV_PANELS_INT]
        test_dataset = features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=TEST_PANEL_INT, freq=freq, type=type, beta_constant=beta)
        datasets_by_fold = None
    else:
        cv_datasets = [SimpleNamespace(panel_number=p) for p in CV_PANELS_INT]
        test_dataset = SimpleNamespace(panel_number=TEST_PANEL_INT)
        datasets_by_fold = {
            val_panel: (
                [Panel_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=p, freq=freq, big_latent=big_latent, type=type, beta_constant=beta, fold=val_panel) for p in CV_PANELS_INT]
                + [Panel_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=TEST_PANEL_INT, freq=freq, big_latent=big_latent, type=type, beta_constant=beta, fold=val_panel)]
            )
            for val_panel in CV_PANELS_INT
        }

    ensemble_predict(cv_datasets, test_dataset, model_path=out_dir,
                      save_dir=os.path.join(out_dir, "ensemble_HI_plot.svg"),
                      datasets_by_fold=datasets_by_fold)

    ensemble_path = os.path.join(out_dir, "ensemble_model.pt")
    build_and_save_ensemble(out_dir, ensemble_path)

    if do_BO:
        # ── BO diagnostic plots ─────────────────────
        fitness_h, damage_h, train_h, val_h = _collect_metric_histories(tuner)
        objective_h = [t.score for t in tuner.oracle.trials.values() if t.score is not None]
        df = _build_trials_dataframe(tuner)
        print(df.head(10))

        plot_optimization_progress(fitness_h, damage_h, f"{out_dir}/bayesian_optimization_progress.svg")
        plot_running_best(objective_h, f"{out_dir}/bayesian_running_best.svg")
        plot_overfitting(df, f"{out_dir}/overfitting_analysis.svg")
        plot_hp_sensitivity(df, f"{out_dir}/hp_sensitivity.svg")
        plot_hp_coverage(df, f"{out_dir}/hp_coverage.svg")

        return {
            "tuner": tuner,
            "best": best,
            "best_params": best_params,
            "fold_checkpoints": fold_net_names,
            "ensemble_path": ensemble_path,
            "fitness_h": fitness_h,
            "damage_h": damage_h,
            "train_h": train_h,
            "val_h": val_h,
            "objective_h": objective_h,
            "df": df,
            "out_dir": out_dir,
        }
    else:
        return {
            "best_params": best_params,
            "fold_checkpoints": fold_net_names,
            "ensemble_path": ensemble_path,
            "out_dir": out_dir,
        }

if __name__ == "__main__":
    run_bayesian_optimization()
