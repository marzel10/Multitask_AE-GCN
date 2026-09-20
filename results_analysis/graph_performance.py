'''
Same analysis as path_performance.py (HI grid, damage map grid, prognostic-criteria
metrics), but for the 6 per-frequency GCN Bayesian-optimization results folders
(BO_GCN.py's output, results/Bayesian_GCN_freq0 .. freq5) instead of the 6
per-path-per-frequency AE folders.
'''

import gc
import pickle
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "tools", "training", "intermediate_results_check", "results_analysis"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch_geometric
import matplotlib.pyplot as plt

from graph_dataset import Panel_GraphDataset, features_GraphDataset
from imagining_alghoritm import P, U, WCPDI
from plot_panel import _draw_static_panel
from prognostic_criteria import monotonicity_criterion, trendability_criterion, prognosability_criterion
from config import (
    GRAPH_DATA_DIR, BO_SEARCH_RESULTS_DIR, BASE_PANELS, FREQUENCY_MAPPING,
     CMAP_HEATMAP, PANEL_W, PANEL_H, FOLD_KEYS, _LINESTYLES, CUSTOM_PALETTE as PALETTE, GCN_RESULTS_DIR, DAMAGE_MAP_N_PIXELS, METRIC_NAMES, TEST_PANEL, LIFETIME_FRACTIONS
)

GCN_FREQ_FOLDERS = [f"Bayesian_GCN_freq{i}" for i in range(6)]
PANELS = [int(p) for p in BASE_PANELS] + [int(TEST_PANEL[0])]   # panel axis (size 5)
N_PATHS = 28

OUT_DIR = GCN_RESULTS_DIR

KEY_TO_TITLES = {
    103: "L1-03", 104: "L1-04", 105: "L1-05", 109: "L1-09", 123: "L1-23",
    "103": "L1-03", "104": "L1-04", "105": "L1-05", "109": "L1-09", "ensemble": "Ensemble",
}



def _palette_style(i):
    color = PALETTE[i % len(PALETTE)]
    linestyle = _LINESTYLES[(i // len(PALETTE)) % len(_LINESTYLES)]
    return color, linestyle


def _model_and_transform(fold, checkpoints):
    '''Fold's model + the normalizing transform its own dataset needs.'''
    ck = checkpoints[fold]
    mean, std = ck["feature_mean"], ck["feature_std"]

    def norm_transform(data, mean=mean, std=std):
        data.x = (data.x - mean) / std
        return data

    return ck["model"], norm_transform


def _ensemble_average(collector, checkpoints, datasets_by_fold, panel, device):
    results = []
    for fold, ck in checkpoints.items():
        model = ck["model"].to(device)
        mean, std = ck["feature_mean"], ck["feature_std"]
        ds = datasets_by_fold[fold][panel]

        def norm_transform(data, mean=mean, std=std):
            data.x = (data.x - mean) / std
            return data

        ds.transform = norm_transform
        results.append(collector(model, ds, device))
    return np.mean(np.stack(results, axis=0), axis=0)


def _collect_HI(model, dataset, device):
    '''Graph-level HI for every state in `dataset`, sorted into increasing-state order.'''
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


def _collect_path_out(model, dataset, device):
    '''Per-path (node-level) outputs for every state in `dataset`, sorted into
    increasing-state order. Returns (n_states, N_PATHS)'''
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=32, shuffle=False)
    all_out, all_states = [], []
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _ = model(data.x, data.edge_index, data.batch, data.edge_weight)
            batch_size = data.num_graphs
            num_paths = out.shape[0] // batch_size
            all_out.append(out.reshape(batch_size, num_paths).cpu().numpy())
            all_states.append(data.y.cpu().numpy().reshape(-1))
    out_arr = np.concatenate(all_out, axis=0)
    states = np.concatenate(all_states)
    return out_arr[np.argsort(states)]


def compute_damage_map_grid(checkpoints, datasets_by_fold, device, n_pixels=DAMAGE_MAP_N_PIXELS):
    '''{panel: {life_fraction: WCPDI map}} averaged over fold sub-models, each evaluated
    on its own fold-matched dataset (see _ensemble_average), at the 5 saved states nearest
    LIFETIME_FRACTIONS. Uses imagining_alghoritm's non-differentiable WCPDI.'''
    dA = (PANEL_W * PANEL_H) / n_pixels
    dx = np.sqrt(dA)
    x = np.arange(0, PANEL_W + dx, dx)
    y = np.arange(0, PANEL_H + dx, dx)
    X, Y = np.meshgrid(x, y, indexing="ij")

    maps = {}
    for panel in PANELS:
        ref_ds = next(iter(datasets_by_fold.values()))[panel]
        ref_ds.transform = None
        n_states = len(ref_ds)
        maps[panel] = {}
        for frac in LIFETIME_FRACTIONS:
            state = int(round(frac * (n_states - 1)))
            U_arr = np.zeros_like(X)
            U(U_arr, (X, Y), panel_number=ref_ds.panel_number, state=state)

            fold_wcpdi = []
            for fold, ck in checkpoints.items():
                model = ck["model"].to(device)
                model.eval()
                mean, std = ck["feature_mean"], ck["feature_std"]
                ds = datasets_by_fold[fold][panel]

                def norm_transform(data, mean=mean, std=std):
                    data.x = (data.x - mean) / std
                    return data

                ds.transform = norm_transform
                P_arr = np.zeros_like(X)
                P(P_arr, (X, Y), state, ds, model)
                fold_wcpdi.append(WCPDI(P_arr, U_arr))

            wcpdi = np.mean(np.stack(fold_wcpdi, axis=0), axis=0)
            maps[panel][frac] = wcpdi
            print(f"  [damage map] panel={panel} life={frac:.0%} state={state}: done")
    return maps


def compute_all(folders=GCN_FREQ_FOLDERS, raw_features=False, big_latent=True, type="peak"):
    '''Builds the HI array (fold x freq x panel -> 1-D array, state order), the metrics
    array (freq x path x metric, ensemble's per-path outputs only), and the damage-map
    grids (one per frequency, ensemble model only, panel x lifetime fraction).'''
    HI = [[[None] * len(PANELS) for _ in folders] for _ in FOLD_KEYS]
    HI_metrics = np.full((len(FOLD_KEYS), len(folders) + 1, len(METRIC_NAMES)), np.nan)  # +1 for the average-over-frequency slot
    metrics = np.full((len(folders) + 1, N_PATHS, len(METRIC_NAMES)), np.nan)
    damage_maps = [None] * (len(folders) + 1)  # +1 for the average-over-frequency slot

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for freq_idx, folder_name in enumerate(folders):
        folder = BO_SEARCH_RESULTS_DIR / folder_name
        if not folder.exists():
            print(f"[{folder_name}] not trained yet, skipping")
            continue

        print(f"[{folder_name}] loading models")
        try:
            missing = [p for p in BASE_PANELS if not (folder / f"gcn_bo_val_{p}.pt").exists()]
            if missing:
                print(f"  missing fold checkpoints {missing}, skipping")
                continue
            ensemble_path = folder / "ensemble_model.pt"
            if not ensemble_path.exists():
                print("  ensemble_model.pt missing, skipping")
                continue

            checkpoints = {
                panel: torch.load(folder / f"gcn_bo_val_{panel}.pt", map_location=device, weights_only=False)
                for panel in BASE_PANELS
            }

            if raw_features:
                shared_datasets = {
                    panel: features_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=panel, freq=freq_idx, type=type)
                    for panel in PANELS
                }
                # no per-fold AE latent space for raw features -- every fold shares the same dataset
                datasets_by_fold = {fold: shared_datasets for fold in BASE_PANELS}
            else:
                # each leave-one-out fold's model was trained on the AE latent space that
                # excluded its own val panel -- evaluate it on that same fold-matched space,
                # not a pooled one (see _ensemble_average).
                datasets_by_fold = {
                    fold: {
                        panel: Panel_GraphDataset(root=str(GRAPH_DATA_DIR), panel_number=panel, freq=freq_idx, big_latent=big_latent, type=type, fold=int(fold))
                        for panel in PANELS
                    }
                    for fold in BASE_PANELS
                }

            # HI grid data: every fold x every panel. 'ensemble' = per-fold models averaged,
            # each on its own fold-matched dataset, rather than the wrapped EnsembleGCN on a
            # single shared input.
            for fold_idx, fold in enumerate(FOLD_KEYS):
                if fold == "ensemble":
                    for panel_idx, panel in enumerate(PANELS):
                        HI[fold_idx][freq_idx][panel_idx] = _ensemble_average(_collect_HI, checkpoints, datasets_by_fold, panel, device)
                else:
                    model, transform = _model_and_transform(fold, checkpoints)
                    model = model.to(device)
                    for panel_idx, panel in enumerate(PANELS):
                        ds = datasets_by_fold[fold][panel]
                        ds.transform = transform
                        HI[fold_idx][freq_idx][panel_idx] = _collect_HI(model, ds, device)

                # Compute HI metrics (Fitness, Mo, Pr, Tr) for every fold and frequency
                Mo = monotonicity_criterion(HI[fold_idx][freq_idx][:])
                Tr = trendability_criterion(HI[fold_idx][freq_idx][:])
                Pr = prognosability_criterion(HI[fold_idx][freq_idx][:])
                HI_metrics[fold_idx][freq_idx] = [Mo + Tr + Pr, Mo, Pr, Tr]

            # Per-path metrics: ensemble's per-path node outputs (per-fold averaged), jointly across panels.
            path_out = {
                panel: _ensemble_average(_collect_path_out, checkpoints, datasets_by_fold, panel, device)
                for panel in PANELS
            }
            for path_i in range(N_PATHS):
                DI = [path_out[panel][:, path_i] for panel in PANELS]
                Mo = monotonicity_criterion(DI)
                Tr = trendability_criterion(DI)
                Pr = prognosability_criterion(DI)
                metrics[freq_idx, path_i] = [Mo + Tr + Pr, Mo, Pr, Tr]

            # Damage map grid for this frequency: ensemble (per-fold averaged).
            damage_maps[freq_idx] = compute_damage_map_grid(checkpoints, datasets_by_fold, device)

            del checkpoints
        except Exception as e:
            print(f"[{folder_name}] FAILED ({e!r})")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for fold_idx, fold in enumerate(FOLD_KEYS):
        avg_curves = [average_HI_over_frequency(HI, panel, fold=fold, folders=folders) for panel in PANELS]
        avg_curves = [c for c in avg_curves if c is not None]
        if not avg_curves:
            continue
        Mo = monotonicity_criterion(avg_curves)
        Tr = trendability_criterion(avg_curves)
        Pr = prognosability_criterion(avg_curves)
        HI_metrics[fold_idx][-1] = [Mo + Tr + Pr, Mo, Pr, Tr]

    damage_maps[len(folders)] = _average_damage_maps_over_frequency(damage_maps[:len(folders)])
    metrics[len(folders)] = np.nanmean(metrics[:len(folders)], axis=0)
    return HI, HI_metrics, metrics, damage_maps


def average_HI_over_frequency(HI, panel, fold="ensemble", folders=GCN_FREQ_FOLDERS):
    '''HI curve for one panel/fold, averaged over the 6 raw frequencies'''
    fold_idx = FOLD_KEYS.index(fold)
    panel_idx = PANELS.index(panel)
    curves = [HI[fold_idx][freq_idx][panel_idx] for freq_idx in range(len(folders))]
    curves = [c for c in curves if c is not None]
    return np.mean(np.stack(curves, axis=0), axis=0) if curves else None


def _average_damage_maps_over_frequency(per_freq_maps):
    '''{panel: {life_fraction: WCPDI map}} averaged over frequencies'''
    valid = [m for m in per_freq_maps if m is not None]
    if not valid:
        return None
    return {
        panel: {
            frac: np.mean(np.stack([m[panel][frac] for m in valid], axis=0), axis=0)
            for frac in LIFETIME_FRACTIONS
        }
        for panel in PANELS
    }


def plot_metrics(metrics, out_dir=OUT_DIR, folders=GCN_FREQ_FOLDERS):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = np.arange(N_PATHS)
    freq_labels = [f"freq {i + 1}" for i in range(len(folders))] + ["average"]

    for metric_idx, metric_name in enumerate(METRIC_NAMES):
        fig, ax = plt.subplots(figsize=(12, 6))
        for freq_idx, label in enumerate(freq_labels):
            is_avg = label == "average"
            color, linestyle = _palette_style(freq_idx)
            ax.plot(paths, metrics[freq_idx, :, metric_idx], marker="o", markersize=3,
                    color="k" if is_avg else color, linestyle="--" if is_avg else linestyle, label=label)
        ax.set_xlabel("Path number")
        ax.set_ylabel(metric_name)
        #ax.set_title(f"{metric_name} vs Path (ensemble model)")
        ax.set_xticks(paths)
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric_name}_vs_path.svg")
        plt.close(fig)
        print(f"Saved: {out_dir / f'{metric_name}_vs_path.svg'}")



def plot_HI_grid(HI, out_dir=OUT_DIR, folders=GCN_FREQ_FOLDERS, weights=None):
    '''One figure, 7x5 grid of subplots: rows are frequencies (6 raw, labeled with their
    actual value from config.FREQUENCY_MAPPING, + average over frequencies), columns are
    folds (4 leave-one-out folds + ensemble). Each subplot (i, j) has one line per panel
    (that panel's graph-level HI), plotted against normalized life length (0-1).'''
    out_dir.mkdir(parents=True, exist_ok=True)
    freq_labels = [f"{FREQUENCY_MAPPING[i]} kHz" for i in range(len(folders))] + ["WAE"]

    fig, axes = plt.subplots(len(freq_labels), len(FOLD_KEYS), figsize=(4 * len(FOLD_KEYS), 3 * len(freq_labels)), sharex=True)

    # per_freq_curves[fold_idx][panel_idx] accumulates that fold/panel's per-frequency
    # curves, to average into the "average over frequencies" row.
    per_freq_curves = [[[] for _ in PANELS] for _ in FOLD_KEYS]

    if weights is None:
        weights = np.ones((len(FOLD_KEYS), len(folders))) / len(folders)

    for freq_idx in range(len(folders)):
        for fold_idx, fold_key in enumerate(FOLD_KEYS):
            ax = axes[freq_idx, fold_idx]
            any_data = False
            WAE_weight = weights[fold_idx][freq_idx]
            for panel_idx, panel in enumerate(PANELS):
                curve = HI[fold_idx][freq_idx][panel_idx]
                if curve is None:
                    continue
                life_fraction = np.linspace(0, 1, curve.shape[0])
                color, linestyle = _palette_style(panel_idx)
                ax.plot(life_fraction, curve, color=color, linestyle=linestyle, label=KEY_TO_TITLES.get(panel, str(panel)))
                per_freq_curves[fold_idx][panel_idx].append(WAE_weight * curve)
                any_data = True
            if not any_data:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=7)
            if freq_idx == 0:
                ax.set_title(KEY_TO_TITLES.get(fold_key, fold_key), fontsize=9)
            if fold_idx == 0:
                ax.set_ylabel(freq_labels[freq_idx], fontsize=9)
            ax.grid(True)


    # WAE row: per fold/panel, mean of its per-frequency curves.
    avg_row = len(folders)
    for fold_idx, fold_key in enumerate(FOLD_KEYS):
        ax = axes[avg_row, fold_idx]
        any_data = False
        for panel_idx, panel in enumerate(PANELS):
            curves = per_freq_curves[fold_idx][panel_idx]
            if not curves:
                continue
            overall_mean = np.sum(np.stack(curves, axis=0), axis=0)
            life_fraction = np.linspace(0, 1, overall_mean.shape[0])
            color, linestyle = _palette_style(panel_idx)
            ax.plot(life_fraction, overall_mean, color=color, linestyle=linestyle, label=KEY_TO_TITLES.get(panel, str(panel)))
            any_data = True
        if not any_data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        if fold_idx == 0:
            ax.set_ylabel(freq_labels[-1], fontsize=9)
        ax.grid(True)

    for ax in axes[-1, :]:
        ax.set_xlabel("Life fraction")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=12, loc="lower center",
               bbox_to_anchor=(0.5, 0.01), ncol=len(PANELS))


    fig.tight_layout(rect=[0, 0.025, 1, 1])
    save_path = out_dir / "HI_grid.svg"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_damage_map_grid(maps, label, file_tag, out_dir=OUT_DIR):
    '''  5x5 grid of raw (non-normalised) WCPDI damage maps -- one row per panel, one column per lifetime fraction, each subplot
    autoscaled to its own data range and given its own colorbar '''
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows, n_cols = len(PANELS), len(LIFETIME_FRACTIONS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)



    for row, panel in enumerate(PANELS):
        for col, frac in enumerate(LIFETIME_FRACTIONS):
            ax = axes[row][col]
            m = maps[panel][frac]

            _draw_static_panel(ax, panel)
            im = ax.imshow(m.T, extent=(0, PANEL_W, 0, PANEL_H), origin="lower", cmap=CMAP_HEATMAP)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="WCPDI Value")

            if row == 0:
                ax.set_title(f"{frac:.0%} lifetime")
            if col == 0:
                ax.set_ylabel(f"Panel {panel}", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])

    #fig.suptitle(f"WCPDI vs lifetime (ensemble model, {label})")
    fig.tight_layout()
    save_path = out_dir / f"damage_map_grid_{file_tag}.svg"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def save_HI_metrics_xlsx(HI_metrics, out_dir=OUT_DIR, folders=GCN_FREQ_FOLDERS):
    '''Writes HI_metrics (fold x frequency x metric) to a tidy .xlsx: one row per
    (fold, frequency), columns fold/frequency/Fitness/Mo/Pr/Tr.'''
    freq_labels = [f"{FREQUENCY_MAPPING[i]} kHz" for i in range(len(folders))] + ["average"]
    rows = []
    for fold_idx, fold in enumerate(FOLD_KEYS):
        for freq_idx, freq_label in enumerate(freq_labels):
            row = {"fold": KEY_TO_TITLES.get(fold, fold), "frequency": freq_label}
            row.update(zip(METRIC_NAMES, HI_metrics[fold_idx, freq_idx]))
            rows.append(row)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "HI_metrics.xlsx"
    df.to_excel(out_path, index=False)
    print(f"Saved: {out_path}")
    return df


def load_cached(out_dir=OUT_DIR):
    '''Loads a previously saved (HI, HI_metrics, metrics, damage_maps) quadruple from out_dir'''
    hi_path = out_dir / "HI.pkl"
    hi_metrics_path = out_dir / "HI_metrics.pkl"
    metrics_path = out_dir / "metrics.npy"
    damage_maps_path = out_dir / "damage_maps.pkl"
    if not hi_path.exists() or not hi_metrics_path.exists() or not metrics_path.exists() or not damage_maps_path.exists():
        return None
    with open(hi_path, "rb") as f:
        HI = pickle.load(f)
    with open(hi_metrics_path, "rb") as f:
        HI_metrics = pickle.load(f)
    metrics = np.load(metrics_path)
    with open(damage_maps_path, "rb") as f:
        damage_maps = pickle.load(f)
    return HI, HI_metrics, metrics, damage_maps


def main(recompute=False, folders=GCN_FREQ_FOLDERS, out_dir=OUT_DIR, raw_features=False, big_latent=True, type="peak"):
    cached = None if recompute else load_cached(out_dir=out_dir)
    if cached is not None:
        print(f"Loaded cached results from {out_dir}")
        HI, HI_metrics, metrics, damage_maps = cached
    else:
        HI, HI_metrics, metrics, damage_maps = compute_all(folders=folders, raw_features=raw_features, big_latent=big_latent, type=type)

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "HI.pkl", "wb") as f:
            pickle.dump(HI, f)
        print(f"Saved: {out_dir / 'HI.pkl'}")

        with open(out_dir / "HI_metrics.pkl", "wb") as f:
            pickle.dump(HI_metrics, f)
        print(f"Saved: {out_dir / 'HI_metrics.pkl'}")

        save_HI_metrics_xlsx(HI_metrics, out_dir=out_dir, folders=folders)

        np.save(out_dir / "metrics.npy", metrics)
        print(f"Saved: {out_dir / 'metrics.npy'}")

        with open(out_dir / "damage_maps.pkl", "wb") as f:
            pickle.dump(damage_maps, f)
        print(f"Saved: {out_dir / 'damage_maps.pkl'}")

    fitness = HI_metrics[:, :, 0]
    WAE_weights = fitness[:,:-1] / np.nansum(fitness[:,:-1], axis=1, keepdims=True)
    plot_metrics(metrics, out_dir=out_dir, folders=folders)
    plot_HI_grid(HI, out_dir=out_dir, folders=folders, weights=WAE_weights)
    for freq_idx, maps in enumerate(damage_maps[:len(folders)]):
        if maps is not None:
            plot_damage_map_grid(maps, f"{FREQUENCY_MAPPING[freq_idx]} kHz", f"freq{freq_idx}", out_dir=out_dir)
    avg_maps = damage_maps[len(folders)]
    if avg_maps is not None:
        plot_damage_map_grid(avg_maps, "average", "average", out_dir=out_dir)


if __name__ == "__main__":
    main(recompute="--recompute" in sys.argv)
