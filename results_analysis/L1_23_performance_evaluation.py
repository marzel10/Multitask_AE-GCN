'''
This file evaluates performance of the CAE and GCN on the unseen new damage type, that
is the data from panel 123 (L1-23) 

For each of the 4 leave-one-out "test-panel models" (test_103_wo123, test_104_wo123,
test_105_wo123, test_109_wo123) and each of the 3 published model types -- "peak"
(CAE-GCN), "raw" (GCN), "path" (CAE) -- this
(re)builds whatever panel-123 input each model type needs and runs that test-panel
model's already-trained ensemble on it:
  - "raw": features_GraphDataset(panel_number=123, ...) is built straight from the
    already-cached raw feature .npy files 
  - "peak": Panel_GraphDataset(panel_number=123, ...) needs AE-latent "shi_raw" files that
    will be extracted in this script if they don't aleady exist (see _ensure_panel123_ae_latent)
  - "path": reuses create_datastores.prepare_datastores, whose ds_dict already includes
    a pooled "123" entry regardless of what panel names are requested, and predicts
    with each path/frequency's saved ensemble_model.keras.

Every (test-panel model, type) result is cached to L1_23_results_analysis/ so a
second run only recomputes what's missing (recompute=True forces a full rebuild).

Task 1: WAE HI curve for panel 123 -- one plot per type (peak/raw/path) and test panel (3x4 grid), one line per
        every panel, combined across frequency with that (test-panel model, type)'s
        own already-cached fitness-based WAE weights (see Compute_WAE.py).
Task 2: prognostic-criteria test metrics (Mo, Pr, Tr, Fitness) treating panel 123 as the
        test panel and each test-panel model's own 3 base (non-test) panels as the CV
        set -- one xlsx table + per-metric plots (vs frequency, one line per test-panel
        model), for every type.
Task 3: WCPDI damage-map grid for panel 123 (rows = test-panel model, columns = life
        fraction, averaged over frequency) -- one figure per type, plus damage maps averaged over test panels results (3(types) x 5(lifefraction) grid)
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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tensorflow as tf
import torch
import torch_geometric

from graph_dataset import Panel_GraphDataset, features_GraphDataset
from extract_shi import extract_shi
from create_datastores import prepare_datastores
from path_performance import _collect_ensemble_DI, CUSTOM_OBJECTS as AE_CUSTOM_OBJECTS
from performance_accross_tests import _normalize_maps
from imagining_alghoritm import P_AE, U, WCPDI
from plot_panel import _draw_static_panel
from prognostic_criteria import monotonicity_criterion, trendability_criterion, prognosability_criterion
from config import (
    PROJECT_ROOT, SINGLE_FILE_PANELS, FREQUENCY_MAPPING, FREQ_LABELS, METRIC_NAMES, TYPES_LABELS,
    LIFETIME_FRACTIONS, DAMAGE_MAP_N_PIXELS, PANEL_W, PANEL_H, CMAP_HEATMAP,
    CUSTOM_PALETTE as PALETTE, _LINESTYLES, SINGLE_FILE_PANELS as TEST_PANEL_MODELS
)

MODEL_TYPES = ["peak", "raw", "path"]
N_PATHS = 28
N_FREQ = len(FREQUENCY_MAPPING)

OUT_DIR = PROJECT_ROOT / "L1_23_results_analysis"
KEY_TO_TITLES = {"103": "L1-03", "104": "L1-04", "105": "L1-05", "109": "L1-09", "123": "L1-23"}


# ─── Path helpers ──────────────────────────────────────────────────────────────
def _test_run_dir(p):
    return PROJECT_ROOT / f"test_{p}_wo123"


def _base_panels(p):
    return [x for x in SINGLE_FILE_PANELS if x != p]


def _graph_data_dir(p):
    return _test_run_dir(p) / "graph_data"


def _bo_results_dir(p):
    return _test_run_dir(p) / "results"


def _model_out_dir(p, model_type):
    if model_type == "path":
        return _test_run_dir(p) / "path_performance_results"
    return _test_run_dir(p) / f"graph_performance_results_{model_type}"


def _palette_style(i):
    color = PALETTE[i % len(PALETTE)]
    linestyle = _LINESTYLES[(i // len(PALETTE)) % len(_LINESTYLES)]
    return color, linestyle


# ─── GCN (peak / raw) panel-123 evaluation ─────────────────────────────────────
def _collect_HI_and_path_out(model, dataset, device):
    '''One forward pass per state, returns both the graph-level HI curve and the
    per-path (node-level) output curve, sorted into increasing-state order.'''
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=32, shuffle=False) # batch size doesn't matter, just for speed
    all_HI, all_out, all_states = [], [], []
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)
            all_HI.append(HI.cpu().numpy().reshape(-1))
            batch_size = data.num_graphs
            num_paths = out.shape[0] // batch_size
            all_out.append(out.reshape(batch_size, num_paths).cpu().numpy())
            all_states.append(data.y.cpu().numpy().reshape(-1))
    HI = np.concatenate(all_HI)
    out_arr = np.concatenate(all_out, axis=0)
    states = np.concatenate(all_states)
    order = np.argsort(states)
    return HI[order], out_arr[order]


def _ensure_panel123_ae_latent(p, freq, fold):
    '''Ensures panel_123_shi_raw_{freq}_fold{fold}.pt exists under this test-panel
    model's graph_data/raw/ -- extracted (once) via extract_shi.extract_shi, mirroring
    extract_shi.pre_compute_AE_output's per-panel loop body, just for dataset="123".'''
    raw_dir = _graph_data_dir(p) / "raw"
    raw_path = raw_dir / f"panel_123_shi_raw_{freq}_fold{fold}.pt"
    if raw_path.exists():
        return

    folders = [f"Multi_path_BO_fixed_freq{freq}\\Bayesian_CNN_AE_path{i}" for i in range(N_PATHS)]
    latents_all, _path_labels, big_latent_all = extract_shi(
        folders, freq, dataset="123", fold=fold, GAN_dir=str(_test_run_dir(p))
    )
    out_dict = {"shi": np.array(latents_all), "path_labels": np.array(_path_labels), "big_latent": np.array(big_latent_all)}
    raw_dir.mkdir(parents=True, exist_ok=True)
    torch.save(out_dict, raw_path)
    print(f"[peak][{p}] extracted AE latent for panel 123, freq={freq}, fold={fold} -> {raw_path}")


def _compute_gcn_panel123(p, model_type, device):
    '''Ensemble (per-CV-fold-averaged) HI curve and per-path output curve for panel 123,
    for every frequency, for one (test-panel model, "peak"/"raw") combination.'''
    base_panels_p = _base_panels(p)
    adjacency_type = "peak" if model_type == "raw" else model_type

    HI_by_freq, path_out_by_freq = [], []
    for freq in range(N_FREQ):
        folder = _bo_results_dir(p) / f"Bayesian_GCN_{model_type}_freq{freq}"
        checkpoints = {}
        for fold in base_panels_p:
            ckpt_path = folder / f"gcn_bo_val_{fold}.pt"
            if not ckpt_path.exists():
                print(f"[{model_type}][{p}] freq={freq}: missing checkpoint {ckpt_path}, skipping frequency")
                checkpoints = None
                break
            checkpoints[fold] = torch.load(ckpt_path, map_location=device, weights_only=False)

        if not checkpoints:
            HI_by_freq.append(None)
            path_out_by_freq.append(None)
            continue

        fold_HI, fold_out = [], []
        shared_ds = None
        if model_type == "raw":
            shared_ds = features_GraphDataset(root=str(_graph_data_dir(p)), panel_number=123, freq=freq, type=adjacency_type)

        for fold, ck in checkpoints.items():
            if model_type == "raw":
                ds = shared_ds
            else:
                _ensure_panel123_ae_latent(p, freq, int(fold))
                ds = Panel_GraphDataset(root=str(_graph_data_dir(p)), panel_number=123, freq=freq,
                                         big_latent=True, type=adjacency_type, fold=int(fold))
            mean, std = ck["feature_mean"], ck["feature_std"]

            def norm_transform(data, mean=mean, std=std):
                data.x = (data.x - mean) / std
                return data
            ds.transform = norm_transform

            model = ck["model"].to(device)
            HI_curve, path_out = _collect_HI_and_path_out(model, ds, device)
            fold_HI.append(HI_curve)
            fold_out.append(path_out)

        HI_by_freq.append(np.mean(np.stack(fold_HI, axis=0), axis=0))
        path_out_by_freq.append(np.mean(np.stack(fold_out, axis=0), axis=0))
        print(f"[{model_type}][{p}] freq={freq}: done ({len(checkpoints)} folds)")

        del checkpoints
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return HI_by_freq, path_out_by_freq


# ─── Path AE panel-123 evaluation ──────────────────────────────────────────────
def _compute_ae_panel123(p):
    '''Ensemble sHI curve (mean over the 28 paths) and per-path sHI array for panel 123,
    for every frequency, for the per-path AE ensemble ("path" type).'''
    base_panels_p = _base_panels(p)
    held_out = base_panels_p[0]  # any fold's own (ds_dict, norm_stats) pair works -- see path_performance._collect_ensemble_DI's docstring
    train_ds_names = [x for x in base_panels_p if x != held_out]

    HI_by_freq, path_out_by_freq = [], []
    for freq in range(N_FREQ):
        per_path = [None] * N_PATHS
        for path_i in range(N_PATHS):
            path_dir = _test_run_dir(p) / f"Multi_path_BO_fixed_freq{freq}" / f"Bayesian_CNN_AE_path{path_i}"
            ensemble_path = path_dir / "ensemble_model.keras"
            if not ensemble_path.exists():
                print(f"[path][{p}] freq={freq} path={path_i}: ensemble_model.keras missing, skipping")
                continue

            _, _, _, ds_dict, *_rest = prepare_datastores(
                path_i=path_i, freq_i=freq, base_batch_size=16, test_batch_size=1,
                train_ds_names=train_ds_names, val_ds_names=[held_out], test_ds_names=[p],
                include_benchmark=True,
            )
            norm_stats = _rest[-1]

            tf.keras.backend.clear_session()
            ensemble_model = tf.keras.models.load_model(str(ensemble_path), custom_objects=AE_CUSTOM_OBJECTS, compile=False)
            DI = _collect_ensemble_DI(ensemble_model, ds_dict, norm_stats, ["123"])
            per_path[path_i] = DI[0].reshape(-1)

            del ensemble_model
            gc.collect()

        valid = [c for c in per_path if c is not None]
        if not valid:
            HI_by_freq.append(None)
            path_out_by_freq.append(None)
            print(f"[path][{p}] freq={freq}: no paths available, skipping")
            continue

        n_states = valid[0].shape[0]
        path_out = np.stack([c if c is not None else np.full(n_states, np.nan) for c in per_path], axis=1)  # (n_states, N_PATHS)
        HI_by_freq.append(np.nanmean(path_out, axis=1))
        path_out_by_freq.append(path_out)
        print(f"[path][{p}] freq={freq}: done ({len(valid)}/{N_PATHS} paths)")

    return HI_by_freq, path_out_by_freq


# ─── Caching ───────────────────────────────────────────────────────────────────
def _cache_path(p, model_type):
    return OUT_DIR / f"_cache_panel123_{model_type}_{p}.pkl"


def _load_or_compute_panel123(p, model_type, device, recompute=False):
    cache_path = _cache_path(p, model_type)
    if cache_path.exists() and not recompute:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"=== Computing panel-123 outputs: type={model_type} test-panel-model={p} ===")
    if model_type == "path":
        HI_by_freq, path_out_by_freq = _compute_ae_panel123(p)
    else:
        HI_by_freq, path_out_by_freq = _compute_gcn_panel123(p, model_type, device)

    result = {"HI_by_freq": HI_by_freq, "path_out_by_freq": path_out_by_freq}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved: {cache_path}")
    return result


# ─── Task 1: WAE HI vs life fraction ────────────────────────────────────────────
PANEL_LABELS = SINGLE_FILE_PANELS + ["123"]  # every panel a subplot's lines cover: the 4 leave-one-out panels + the unseen-damage panel this whole file evaluates


def _panel_HI_by_freq(p, model_type):
    '''Ensemble-fold HI curves {panel: [one curve per raw frequency]} for the 3 base panels
    plus panel p itself, read from (test-panel model p, type)'s already-cached HI.pkl.
    HI.pkl's panel axis is _base_panels(p) + [p] (see graph_performance.py / path_performance.py).'''
    panel_axis = _base_panels(p) + [p]
    with open(_model_out_dir(p, model_type) / "HI.pkl", "rb") as f:
        HI = pickle.load(f)  # fold x freq x panel -> 1D array (state order)
    return {panel: [HI[-1][freq][i] for freq in range(N_FREQ)] for i, panel in enumerate(panel_axis)}

def compute_WAE_weights(results, model_type, p):
    '''compute fitness for all panels including L1-23 to later compute WAE weights'''
   
    by_panel = _panel_HI_by_freq(p, model_type)
    base = _base_panels(p)
    HI_123_by_freq = results[model_type][p]["HI_by_freq"]

    per_freq_metrics = []
    for freq in range(N_FREQ):
        if HI_123_by_freq[freq] is None:
            per_freq_metrics.append(None)
            continue
        DI = [by_panel[b][freq] for b in base] + [HI_123_by_freq[freq]]  # 3 base panels + L1-23 (test)
        test_idx = len(DI) - 1
        Mo = monotonicity_criterion(DI)
        Pr = prognosability_criterion(DI)
        Tr = trendability_criterion(DI)
        per_freq_metrics.append([Mo + Pr + Tr, Mo, Pr, Tr])

    per_freq_metrics = np.array(per_freq_metrics)
    print(f"fitness for {model_type} {p}: {per_freq_metrics[:, 0]}")    

    fitness = per_freq_metrics[:, 0] # every frequency
    weights = fitness / np.sum(fitness)
    return weights

def task1_WAE_HI(results):
    '''One figure, 3x4 subplots -- rows are model_type (peak/raw/path), columns are
    test-panel model -- each with one WAE HI line per panel in PANEL_LABELS'''
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(MODEL_TYPES), len(TEST_PANEL_MODELS),
                              figsize=(7 * len(TEST_PANEL_MODELS), 5 * len(MODEL_TYPES)), squeeze=False)
    row_to_type = {"path":0,"peak":1,"raw":2}
    for row, model_type in enumerate(MODEL_TYPES):
        row = row_to_type[model_type]
        for col, p in enumerate(TEST_PANEL_MODELS):
            ax = axes[row][col]
            by_panel = _panel_HI_by_freq(p, model_type)
            by_panel["123"] = results[model_type][p]["HI_by_freq"]
            weights = compute_WAE_weights(results, model_type, p)
            
            any_line = False
            for i, panel in enumerate(PANEL_LABELS):
                HI_by_freq = by_panel.get(panel)
                if HI_by_freq is None or any(c is None for c in HI_by_freq):
                    print(f"[task1][{model_type}][{p}][{panel}] missing some frequencies, skipping line")
                    continue
                WAE_HI = np.sum([w * c for w, c in zip(weights, HI_by_freq)], axis=0)
                life_fraction = np.linspace(0, 1, WAE_HI.shape[0])
                color, linestyle = _palette_style(i)
                ax.plot(life_fraction * 100, WAE_HI, color=color, linestyle=linestyle, label=KEY_TO_TITLES.get(panel, panel))
                any_line = True
            if not any_line:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=15)
            if row == 0:
                ax.set_title(KEY_TO_TITLES.get(p, p), fontsize=20)
            if col == 0:
                ax.set_ylabel(f"{TYPES_LABELS.get(model_type, model_type)}\n", fontsize=20)
            if row == len(MODEL_TYPES) - 1:
                ax.set_xlabel("Lifetime (%)", fontsize=20)
            ax.tick_params(axis="both", labelsize=16)
            ax.grid(True)

    legend_handles = [
        Line2D([0], [0], color=_palette_style(i)[0], linestyle=_palette_style(i)[1], label=KEY_TO_TITLES.get(panel, panel))
        for i, panel in enumerate(PANEL_LABELS)
    ]
    fig.legend(handles=legend_handles, fontsize=20, title_fontsize=20,
               loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(legend_handles))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_path = OUT_DIR / "WAE_HI_L123.svg"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ─── Task 2: L1-23-as-test prognostic-criteria metrics ─────────────────────────
def _summarize_across_test_panel_models(df):
    '''Collapses task2_test_metrics's per-(model_type, test_panel_model, frequency) table
    to mean +/- std of each metric across the 4 test-panel models -- one row per
    (model_type, frequency), plus n_test_panel_models (how many of the 4 actually had a
    value, since a frequency/type combo can be partially missing).'''
    rows = []
    for model_type in MODEL_TYPES:
        for freq_label in FREQ_LABELS:
            sub = df[(df["model_type"] == model_type) & (df["frequency"] == freq_label)]
            if sub.empty:
                continue
            row = {"model_type": model_type, "frequency": freq_label, "n_test_panel_models": len(sub)}
            for metric in METRIC_NAMES:
                row[f"{metric}_mean"] = sub[metric].mean()
                row[f"{metric}_std"] = sub[metric].std(ddof=0)
            rows.append(row)
    return pd.DataFrame(rows)


def task2_test_metrics(results):
    """Computes fitness and stores it in xlxs file and plots the metrics for each model type and test panel model vs frequency"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_type in MODEL_TYPES:
        for p in TEST_PANEL_MODELS:
            by_panel = _panel_HI_by_freq(p, model_type)
            base = _base_panels(p)
            HI_123_by_freq = results[model_type][p]["HI_by_freq"]

            per_freq_metrics = []
            for freq in range(N_FREQ):
                if HI_123_by_freq[freq] is None:
                    per_freq_metrics.append(None)
                    continue
                DI = [by_panel[b][freq] for b in base] + [HI_123_by_freq[freq]]  # 3 base panels + L1-23 (test)
                test_idx = len(DI) - 1
                Mo = monotonicity_criterion(DI, test_mode=True, test_idx=test_idx)
                Pr = prognosability_criterion(DI, test_mode=True, test_idx=test_idx)
                Tr = trendability_criterion(DI)
                per_freq_metrics.append([Mo + Pr + Tr, Mo, Pr, Tr])

            valid_freqs = [freq for freq in range(N_FREQ) if HI_123_by_freq[freq] is not None]
            if valid_freqs:
                weights = compute_WAE_weights(results, model_type, p)
                valid_weights = weights[valid_freqs]
                valid_weights = valid_weights / np.sum(valid_weights)

                avg_base = [
                    np.sum([w * by_panel[b][freq] for w, freq in zip(valid_weights, valid_freqs)], axis=0)
                    for b in base
                ]
                avg_123 = np.sum([w * HI_123_by_freq[freq] for w, freq in zip(valid_weights, valid_freqs)], axis=0)
                DI_avg = avg_base + [avg_123]
                test_idx = len(DI_avg) - 1
                Mo = monotonicity_criterion(DI_avg, test_mode=True, test_idx=test_idx)
                Pr = prognosability_criterion(DI_avg, test_mode=True, test_idx=test_idx)
                Tr = trendability_criterion(DI_avg)
                per_freq_metrics.append([Mo + Pr + Tr, Mo, Pr, Tr])
            else:
                per_freq_metrics.append(None)

            for freq_idx, freq_label in enumerate(FREQ_LABELS):  # 6 raw + "average"
                m = per_freq_metrics[freq_idx]
                if m is None:
                    continue
                row = {"model_type": model_type, "test_panel_model": KEY_TO_TITLES.get(p, p), "frequency": freq_label}
                row.update(zip(METRIC_NAMES, m))
                rows.append(row)

    df = pd.DataFrame(rows)
    summary_df = _summarize_across_test_panel_models(df)

    out_path = OUT_DIR / "L123_test_metrics.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="metrics", index=False)
        summary_df.to_excel(writer, sheet_name="summary_across_test_panels", index=False)
    print(f"Saved: {out_path}")

    x = np.arange(len(FREQ_LABELS))
    for model_type in MODEL_TYPES:
        sub_type = df[df["model_type"] == model_type]
        if sub_type.empty:
            continue
        for metric_name in METRIC_NAMES:
            fig, ax = plt.subplots(figsize=(9, 5))
            any_line = False
            for i, p in enumerate(TEST_PANEL_MODELS):
                label = KEY_TO_TITLES.get(p, p)
                sub = sub_type[sub_type["test_panel_model"] == label].set_index("frequency").reindex(FREQ_LABELS)
                if sub[metric_name].isna().all():
                    continue
                color, linestyle = _palette_style(i)
                ax.plot(x, sub[metric_name].values, marker="o", linestyle=linestyle, color=color, label=label)
                any_line = True
            if not any_line:
                plt.close(fig)
                continue
            ax.set_xticks(x)
            ax.set_xticklabels(FREQ_LABELS, rotation=45, ha="right")
            ax.set_xlabel("Frequency")
            ax.set_ylabel(metric_name)
            ax.set_title(f"L1-23 test metric: {metric_name} ({TYPES_LABELS.get(model_type, model_type)})")
            ax.legend(title="Test-panel model", fontsize=8)
            ax.grid(True)
            fig.tight_layout()
            save_path = OUT_DIR / f"L123_test_metric_{metric_name}_{model_type}.svg"
            fig.savefig(save_path)
            plt.close(fig)
            print(f"Saved: {save_path}")


# ─── Task 3: damage-map grid + scalar damage metric ────────────────────────────
def _wcpdi_maps_for_life_fractions(per_path_out, n_pixels=DAMAGE_MAP_N_PIXELS, fractions=LIFETIME_FRACTIONS):
    ''' dictionary of {life_fraction: WCPDI map} for panel 123 for every frequency,
    from an already-computed (n_states, N_PATHS) per_path_out array'''
    dA = (PANEL_W * PANEL_H) / n_pixels
    dx = np.sqrt(dA)
    x = np.arange(0, PANEL_W + dx, dx)
    y = np.arange(0, PANEL_H + dx, dx)
    X, Y = np.meshgrid(x, y, indexing="ij")

    n_states = per_path_out.shape[0]
    maps = {}
    for frac in fractions:
        state = int(round(frac * (n_states - 1)))
        U_arr = np.zeros_like(X)
        U(U_arr, (X, Y), panel_number=123, state=state)
        P_arr = np.zeros_like(X)
        P_AE(P_arr, (X, Y), per_path_out[state], panel_number=123, state=state)
        maps[frac] = WCPDI(P_arr, U_arr)
    return maps


def _average_maps_over_freq(maps_by_freq):
    return {frac: np.mean(np.stack([m[frac] for m in maps_by_freq], axis=0), axis=0) for frac in LIFETIME_FRACTIONS}


def _plot_damage_map_grid_L123(grid_maps, model_type, normalize=False):
    '''grid_maps: {test_panel_model: {life_fraction: map}}. One figure: rows = test-panel
    model, columns = life fraction. normalize=True rescales each row into [0, 1]
    via performance_accross_tests._normalize_maps, then the whole grid shares one color scale and one
    colorbar below the figure; normalize=False leaves each cell on its own raw scale, so
    each cell autoscales and gets its own colorbar instead'''
    panels_present = [p for p in TEST_PANEL_MODELS if p in grid_maps]
    n_rows, n_cols = len(panels_present), len(LIFETIME_FRACTIONS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    if normalize:
        grid_maps = {p: _normalize_maps(grid_maps[p]) for p in panels_present}
    vmin, vmax = (0.0, 1.0) if normalize else (None, None)

    im = None
    for row, p in enumerate(panels_present):
        for col, frac in enumerate(LIFETIME_FRACTIONS):
            ax = axes[row][col]
            m = grid_maps[p][frac]
            _draw_static_panel(ax, 123)
            im = ax.imshow(m.T, extent=(0, PANEL_W, 0, PANEL_H), origin="lower", cmap=CMAP_HEATMAP, vmin=vmin, vmax=vmax)
            if not normalize:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.set_title(r"$\Psi (-)$", fontsize=20, loc="left", pad=12)
                cbar.set_ticks(np.round(np.linspace(*im.get_clim(), 4)).astype(int))
                cbar.ax.tick_params(labelsize=20)
                cbar.ax.tick_params(labelsize=20)
            if row == 0:
                ax.set_title(f"{frac:.0%} lifetime", fontsize=20)
            if col == 0:
                ax.set_ylabel(KEY_TO_TITLES.get(p, p), fontsize=20)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"L1-23 WCPDI vs lifetime, avg over frequency ({TYPES_LABELS.get(model_type, model_type)})")
    if normalize:
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        if im is not None:
            cax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
            cbar.ax.set_title(r"$\Psi (-)$", fontsize=20, loc="left", pad=12)
            cbar.set_ticks(np.round(np.linspace(*im.get_clim(), 4)).astype(int))
            cbar.ax.tick_params(labelsize=20)
    else:
        fig.tight_layout()

    if normalize:
        save_path = OUT_DIR / f"damage_map_grid_L123_{model_type}.svg"
    else:
        save_path = OUT_DIR / f"damage_map_grid_L123_{model_type}_not_normalized.svg"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def task3_damage_maps(results, normalize=True):
    '''
    renders and plots the WCPDI damage-map grid for panel 123, averaged over frequency, for every test-panel model and every type (peak/raw/path)
    '''
    OUT_DIR.mkdir(parents=True, exist_ok=True)
   
    all_grid_maps = {}

    for model_type in MODEL_TYPES:
        grid_maps = {}
        
        for p in TEST_PANEL_MODELS:
            path_out_by_freq = results[model_type][p]["path_out_by_freq"]
            valid_freqs = [freq for freq in range(N_FREQ) if path_out_by_freq[freq] is not None]
            if not valid_freqs:
                print(f"[task3][{model_type}][{p}] no data, skipping")
                continue

            maps_by_freq, per_freq_diffs = [], []
            for freq in valid_freqs:
                per_path_out = path_out_by_freq[freq]  # (n_states, N_PATHS)
                maps_by_freq.append(_wcpdi_maps_for_life_fractions(per_path_out))

            grid_maps[p] = _average_maps_over_freq(maps_by_freq)

        if grid_maps:
            _plot_damage_map_grid_L123(grid_maps, model_type, normalize=normalize)
        
        all_grid_maps[model_type] = grid_maps


    return all_grid_maps


# ─── Task 4: damage-map grid averaged over every test-panel model ─────────────
def _average_maps_over_test_panels(grid_maps):
    return {
        frac: np.mean(np.stack([grid_maps[p][frac] for p in grid_maps], axis=0), axis=0)
        for frac in LIFETIME_FRACTIONS
    }


def task4_avg_damage_map_grid(all_grid_maps, normalize=True):
    '''One figure: rows = model type (peak/raw/path), columns = life fraction, each cell
    the WCPDI map averaged over frequency (task3) and then over every test-panel model.'''
    ROW_ORDER = ["path", "peak", "raw"]
    types_present = [t for t in ROW_ORDER if all_grid_maps.get(t)]
    if not types_present:
        print("[task4] no damage-map data for any type, skipping")
        return

    n_rows, n_cols = len(types_present), len(LIFETIME_FRACTIONS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    vmin, vmax = (0.0, 1.0) if normalize else (None, None)

    im = None
    for row, model_type in enumerate(types_present):
        avg_maps = _average_maps_over_test_panels(all_grid_maps[model_type])
        if normalize:
            avg_maps = _normalize_maps(avg_maps)  # same normalization as task3_damage_maps's normalize=True branch
        for col, frac in enumerate(LIFETIME_FRACTIONS):
            ax = axes[row][col]
            m = avg_maps[frac]
            _draw_static_panel(ax, 123)
            im = ax.imshow(m.T, extent=(0, PANEL_W, 0, PANEL_H), origin="lower", cmap=CMAP_HEATMAP, vmin=vmin, vmax=vmax)
            if not normalize:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=20)
            if row == 0:
                ax.set_title(f"{frac:.0%} lifetime", fontsize=20)
            if col == 0:
                ax.set_ylabel(TYPES_LABELS.get(model_type, model_type), fontsize=20)
            ax.set_xticks([])
            ax.set_yticks([])

    if normalize:
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        if im is not None:
            cax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal", label="Normalized WCPDI")
            cbar.ax.tick_params(labelsize=20)
        save_path = OUT_DIR / "damage_map_grid_L123_avg_over_test_panels.svg"
    else:
        fig.tight_layout()
        save_path = OUT_DIR / "damage_map_grid_L123_avg_over_test_panels_not_normalized.svg"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ─── Main ────────────────────────────────────────────────────────────────────
def main(recompute=False, model_types=["raw", "path", "peak"], test_panels=TEST_PANEL_MODELS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {model_type: {} for model_type in model_types}
    for model_type in model_types:
        for p in test_panels:
            results[model_type][p] = _load_or_compute_panel123(p, model_type, device, recompute=recompute)

    task1_WAE_HI(results)
    task2_test_metrics(results)
    all_grid_maps = task3_damage_maps(results,normalize=False)
    task4_avg_damage_map_grid(all_grid_maps, normalize=False)


if __name__ == "__main__":
    main(recompute="--recompute" in sys.argv)
