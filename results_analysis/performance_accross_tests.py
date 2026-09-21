'''
This files uses results written by path_performance.py and graph_performance.py 
(for each test panel) to produce a cross-test-panel summary of the results.

Every function here only reads cached artifacts (HI.pkl, HI_metrics.pkl,
HI_test_metrics.pkl, WAE_HI_metrics.pkl, WAE_HI_test_metrics.pkl, damage_maps.pkl,
sHI.pkl, and damage_loss_evaluation_results/damage_metrics_*.pkl) -- no model is ever
reloaded. 
'''
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
from matplotlib.patches import Patch

from imagining_alghoritm import P_AE, U, WCPDI
from plot_panel import _draw_static_panel
from config import (
    PROJECT_ROOT, FREQUENCY_MAPPING, METRIC_COLUMNS, MODEL_TYPES, GCN_TYPES,
    TYPES_LABELS, PANEL_W, PANEL_H, CMAP_HEATMAP, LIFETIME_FRACTIONS, DAMAGE_MAP_N_PIXELS,
    BETA_CONSTANT, CUSTOM_PALETTE as PALETTE, _LINESTYLES, SINGLE_FILE_PANELS as BASE_PANELS_SET
)

ALL_PANELS = BASE_PANELS_SET  # test panels aggregated across in every task below (123 intentionally excluded)
OUT_DIR = PROJECT_ROOT / "test_panel_performance_results"
FREQ_LABELS = [f"{f} kHz" for f in FREQUENCY_MAPPING] + ["WAE"]

KEY_TO_TITLES = {
    "103": "L1-03", "104": "L1-04", "105": "L1-05", "109": "L1-09", "123": "L1-23",
    103: "L1-03", 104: "L1-04", 105: "L1-05", 109: "L1-09", 123: "L1-23",
    "ensemble": "Ensemble",
}


def _palette_style(i):
    color = PALETTE[i % len(PALETTE)]
    linestyle = _LINESTYLES[(i // len(PALETTE)) % len(_LINESTYLES)]
    return color, linestyle


_HI_GRID_COLORS = [PALETTE[0], PALETTE[1], PALETTE[2], "#6F4423"]  # blue, orange, light blue from CUSTOM_PALETTE + brown; one fixed color per panel slot (base panels + test panel) in plot_HI_grid_across_tests


def _base_panels_for(test_panel):
    return [p for p in BASE_PANELS_SET if p != test_panel]


def _panels_for(test_panel):
    return _base_panels_for(test_panel) + [test_panel]


def _last_fold_idx(arr):
    return len(arr) - 1  

def _ensemble_fold_idx(test_panel):
    return len(_base_panels_for(test_panel))  

def _model_dir(test_panel, model_type):
    root = PROJECT_ROOT / f"test_{test_panel}_wo123"
    if model_type == "path":
        return root / "path_performance_results"
    return root / f"graph_performance_results_{model_type}"


def _load_pickle(path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ===========================================================================
# Task 1 -- HI grid across test panels (rows = frequency + WAE, columns = test panel)
# ===========================================================================
def plot_HI_grid_across_tests(model_type, out_dir=OUT_DIR):
    '''
    7x5 figures grid of HI vs lifetime(%). 
    Each subplot has one line per panel, only ensemble model results are shown
    '''
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows, n_cols = len(FREQ_LABELS), len(ALL_PANELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)

    for col, test_panel in enumerate(ALL_PANELS):
        HI = _load_pickle(_model_dir(test_panel, model_type) / "HI.pkl")
        panels = _panels_for(test_panel)
        fold_idx = _last_fold_idx(HI) if HI is not None else None
        n_panel_axis = len(HI[fold_idx][0]) if HI is not None else None
        if HI is None or n_panel_axis != len(panels):
            reason = "no HI.pkl" if HI is None else f"{n_panel_axis} panels on disk, expected {len(panels)} (different CV scheme)"
            print(f"[{model_type}][{test_panel}] {reason}, skipping column")
            for row in range(n_rows):
                axes[row, col].text(0.5, 0.5, "no data", ha="center", va="center",
                                     transform=axes[row, col].transAxes, fontsize=15)
            axes[0, col].set_title(KEY_TO_TITLES.get(test_panel, test_panel), fontsize=15)
            continue

        HI_metrics = _load_pickle(_model_dir(test_panel, model_type) / "HI_metrics.pkl")
        n_freq = len(HI[fold_idx])

        if HI_metrics is not None and _last_fold_idx(HI_metrics) == fold_idx:
            fitness = np.asarray(HI_metrics[fold_idx, :n_freq, 0], dtype=float)
            with np.errstate(invalid="ignore"):
                weights = fitness / np.nansum(fitness)
        else:
            weights = np.full(n_freq, 1.0 / n_freq)

        per_freq_weighted = [[] for _ in panels]
        for freq_idx in range(n_freq):
            ax = axes[freq_idx, col]
            any_data = False
            w = weights[freq_idx] if np.isfinite(weights[freq_idx]) else 0.0
            for panel_idx, panel in enumerate(panels):
                curve = HI[fold_idx][freq_idx][panel_idx]
            
                if curve is None:
                    continue
                life_fraction = np.linspace(0, 100, curve.shape[0])
                color = _HI_GRID_COLORS[panel_idx % len(_HI_GRID_COLORS)]
                ax.plot(life_fraction, curve, color=color, linestyle="-",
                        label=KEY_TO_TITLES.get(panel, panel))
                per_freq_weighted[panel_idx].append(w * curve)
                any_data = True
            if not any_data:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=15)
            if freq_idx == 0:
                ax.set_title(KEY_TO_TITLES.get(test_panel, test_panel), fontsize=15)
            ax.grid(True)

        ax = axes[n_rows - 1, col]
        any_data = False
        for panel_idx, panel in enumerate(panels):
            curves = per_freq_weighted[panel_idx]
            if not curves:
                continue
            overall = np.sum(np.stack(curves, axis=0), axis=0)
            life_fraction = np.linspace(0, 100, overall.shape[0])
            color = _HI_GRID_COLORS[panel_idx % len(_HI_GRID_COLORS)]
            ax.plot(life_fraction, overall, color=color, linestyle="-",
                    label=KEY_TO_TITLES.get(panel, panel))
            any_data = True
        if not any_data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        ax.grid(True)

    for ax in axes[-1, :]:
        ax.set_xlabel("Lifetime (%)", fontsize=15)
    for row in range(n_rows):
        axes[row, 0].set_ylabel(FREQ_LABELS[row], fontsize=15)
    for ax in axes.flat:
        ax.tick_params(axis='both', labelsize=15)

    handles_by_label = {}
    for row in axes:
        for ax in row:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                handles_by_label.setdefault(ll, hh)
    label_order = [KEY_TO_TITLES.get(p, p) for p in ALL_PANELS]
    labels = [l for l in label_order if l in handles_by_label]
    handles = [handles_by_label[l] for l in labels]
    if handles:
        fig.legend(handles, labels, fontsize=15, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=min(len(labels), 5))

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_path = out_dir / f"HI_grid_{model_type}.svg"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ===========================================================================
# Task 2.1 / 2.2 -- HI_metrics.xlsx, sheets "all" and "test"
# ===========================================================================
def _collect_HI_metrics_rows(pkl_filename, wae_filename):
    '''One row per (model_type, test_panel, frequency), ensemble fold only. The 6 raw
    frequencies come from `pkl_filename` (HI_metrics.pkl / HI_test_metrics.pkl); the
    7th row ("WAE") comes from `wae_filename` (WAE_HI_metrics.pkl / WAE_HI_test_metrics.pkl).'''
    rows = []
    for test_panel in ALL_PANELS:
        for model_type in MODEL_TYPES:
            model_dir = _model_dir(test_panel, model_type)

            HI_metrics = _load_pickle(model_dir / pkl_filename)
            if HI_metrics is not None:
                fold_idx = _last_fold_idx(HI_metrics)
                n_freq = HI_metrics.shape[1] - 1  # last slot is the plain average-over-freq, not used here
                for freq_idx in range(n_freq):
                    row = {"model_type": model_type, "test_panel": test_panel, "frequency": FREQUENCY_MAPPING[freq_idx]}
                    row.update(zip(METRIC_COLUMNS, HI_metrics[fold_idx, freq_idx]))
                    rows.append(row)
            else:
                print(f"[{model_type}][{test_panel}] no {pkl_filename}, skipping")

            WAE_metrics = _load_pickle(model_dir / wae_filename)
            if WAE_metrics is not None:
                fold_idx = _last_fold_idx(WAE_metrics)
                row = {"model_type": model_type, "test_panel": test_panel, "frequency": "WAE"}
                row.update(zip(METRIC_COLUMNS, WAE_metrics[fold_idx]))
                rows.append(row)
            else:
                print(f"[{model_type}][{test_panel}] no {wae_filename}, skipping WAE row")
    return pd.DataFrame(rows)


def _summarize_across_test_panels(df):
    '''Collapses a _collect_HI_metrics_rows dataframe (one row per model_type/test_panel/
    frequency) to mean +/- std of each metric across the ALL_PANELS test-panel runs --
    one row per (model_type, frequency), plus n_test_panels (how many runs actually had
    a cached value).'''
    freq_order = list(FREQUENCY_MAPPING) + ["WAE"]
    rows = []
    for model_type in MODEL_TYPES:
        for freq in freq_order:
            sub = df[(df["model_type"] == model_type) & (df["frequency"] == freq)]
            if sub.empty:
                continue
            row = {"model_type": model_type, "frequency": freq, "n_test_panels": len(sub)}
            for metric in METRIC_COLUMNS:
                row[f"{metric}_mean"] = sub[metric].mean()
                row[f"{metric}_std"] = sub[metric].std(ddof=0)
            rows.append(row)
    return pd.DataFrame(rows)


def save_HI_metrics_summary_xlsx(out_dir=OUT_DIR):
    '''Saves HI metrics to an excel file with 4 sheets:
    -"all": all test panels and frequencies 
    -"test"; test metrics for all panels 
    -"all_fitness": mean and std of  metrics across test panels, for each model_type/frequency
    -"test_fitness": mean and std of test metrics across test panels, for each model_type/frequency 
    '''
    out_dir.mkdir(parents=True, exist_ok=True)
    df_all = _collect_HI_metrics_rows("HI_metrics.pkl", "WAE_HI_metrics.pkl")
    df_test = _collect_HI_metrics_rows("HI_test_metrics.pkl", "WAE_HI_test_metrics.pkl")
    summary_all = _summarize_across_test_panels(df_all)
    summary_test = _summarize_across_test_panels(df_test)

    out_path = out_dir / "HI_metrics.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="all", index=False)
        df_test.to_excel(writer, sheet_name="test", index=False)
        summary_all.to_excel(writer, sheet_name="all_fitness", index=False)
        summary_test.to_excel(writer, sheet_name="test_fitness", index=False)
    print(f"Saved: {out_path}")
    return df_all, df_test, summary_all, summary_test


# ===========================================================================
# Task 3 -- damage map grid across test panels (rows = test panel, columns = life fraction)
# ===========================================================================
def _sHI_avg_over_freq(sHI, test_panel, panel):
    '''Per-path sHI curve (length-28 list of 1-D arrays) for one panel, ensemble fold,
    averaged over that run's raw frequencies.'''
    panels = _panels_for(test_panel)
    fold_idx = _last_fold_idx(sHI)
    if len(sHI[fold_idx][0]) != len(panels):
        return None
    panel_idx = panels.index(panel)
    n_freq = len(sHI[fold_idx])
    n_paths = len(sHI[fold_idx][0][panel_idx])
    averaged = []
    for path_i in range(n_paths):
        curves = [sHI[fold_idx][f][panel_idx][path_i] for f in range(n_freq)]
        curves = [c for c in curves if c is not None]
        averaged.append(np.mean(np.stack(curves, axis=0), axis=0) if curves else None)
    return averaged


def _normalize_maps(maps):
    '''normalizes every WCPDI map in `maps` to [0, 1] using the global min/max across all maps.'''
    stacked = np.stack(list(maps.values()), axis=0)
    vmin, vmax = np.nanmin(stacked), np.nanmax(stacked)
    if vmax > vmin:
        return {frac: (m - vmin) / (vmax - vmin) for frac, m in maps.items()}
    return {frac: np.zeros_like(m) for frac, m in maps.items()}


def _damage_map_grid_for_path(sHI, test_panel, panel, life_fractions, n_pixels, normalize=False):
    '''{life_fraction: WCPDI map} for the CAE ("path") ensemble, recomputed 
    from sHI.pkl'''
    sHI_avg = _sHI_avg_over_freq(sHI, test_panel, panel)
    if sHI_avg is None:
        return None
    n_states_seen = {len(c) for c in sHI_avg if c is not None}
    if not n_states_seen:
        return None
    n_states = n_states_seen.pop()
    sHI_avg = [c if c is not None else np.zeros(n_states) for c in sHI_avg]

    dA = (PANEL_W * PANEL_H) / n_pixels
    dx = np.sqrt(dA)
    x = np.arange(0, PANEL_W + dx, dx)
    y = np.arange(0, PANEL_H + dx, dx)
    X, Y = np.meshgrid(x, y, indexing="ij")
    grid = (X, Y)

    maps = {}
    for frac in life_fractions:
        state = int(round(frac * (n_states - 1)))
        U_arr = np.zeros_like(X)
        U(U_arr, grid, panel_number=panel, state=state)
        P_arr = np.zeros_like(X)
        sHI_per_state = [c[state] for c in sHI_avg]
        P_AE(P_arr, grid, sHI_per_state, panel_number=panel, state=state)
        maps[frac] = WCPDI(P_arr, U_arr)

    if normalize:
        maps = _normalize_maps(maps)
    return maps


def plot_damage_map_grid_across_tests(model_type, out_dir=OUT_DIR, n_pixels=DAMAGE_MAP_N_PIXELS,
                                       life_fractions=LIFETIME_FRACTIONS, normalize=False):
    '''One figure per model_type: rows are test panels, columns are life fractions.
    For GCN types, the damage maps are read from damage_maps.pkl (already averaged over frequency, see
    graph_performance.compute_all); "path" is recomputed from sHI.pkl (Task 3).'''
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows, n_cols = len(ALL_PANELS), len(life_fractions)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    maps_by_row = []
    for test_panel in ALL_PANELS:
        panel_int = int(test_panel)
        if model_type == "path":
            sHI = _load_pickle(_model_dir(test_panel, model_type) / "sHI.pkl")
            maps = _damage_map_grid_for_path(sHI, test_panel, test_panel, life_fractions, n_pixels, normalize=normalize) if sHI is not None else None
            if sHI is None:
                print(f"[path][{test_panel}] no sHI.pkl, skipping row")
        else:
            damage_maps = _load_pickle(_model_dir(test_panel, model_type) / "damage_maps.pkl")
            maps = damage_maps[-1].get(panel_int) if damage_maps is not None else None
            if damage_maps is None:
                print(f"[{model_type}][{test_panel}] no damage_maps.pkl, skipping row")
            elif normalize and maps is not None:
                maps = _normalize_maps(maps)
                
        maps_by_row.append(maps)

    vmin, vmax = (0.0, 1.0) if normalize else (None, None)

    im = None
    for row, test_panel in enumerate(ALL_PANELS):
        panel_int = int(test_panel)
        maps = maps_by_row[row]
        for col, frac in enumerate(life_fractions):
            ax = axes[row][col]
            if row == 0:
                ax.set_title(f"{frac:.0%} lifetime", fontsize=20)
            ax.set_xticks([])
            ax.set_yticks([])
            if maps is None or frac not in maps:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=20)
                if col == 0:
                    ax.set_ylabel(KEY_TO_TITLES.get(test_panel, test_panel), fontsize=18)
                continue
            m = maps[frac]
            _draw_static_panel(ax, panel_int)  # sets its own "y (m)" ylabel -- must set ours after, not before
            im = ax.imshow(m.T, extent=(0, PANEL_W, 0, PANEL_H), origin="lower", cmap=CMAP_HEATMAP, vmin=vmin, vmax=vmax)
            if not normalize:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.set_title(r"$\Psi (-)$", fontsize=20, loc="left", pad=12)
                cbar.set_ticks(np.round(np.linspace(*im.get_clim(), 4)).astype(int))
                cbar.ax.tick_params(labelsize=20)
            if col == 0:
                ax.set_ylabel(KEY_TO_TITLES.get(test_panel, test_panel), fontsize=20)

    if normalize:
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        if im is not None:
            cax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
            cbar.ax.set_title(r"$\Psi (-)$", fontsize=20, loc="left", pad=12)
            cbar.set_ticks(np.round(np.linspace(*im.get_clim(), 4)).astype(int))
            cbar.ax.tick_params(labelsize=20)
        save_path = out_dir / f"damage_map_grid_{model_type}.svg"
    else:
        fig.tight_layout()
        save_path = out_dir / f"damage_map_grid_{model_type}_not_normalized.svg"

    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")



# ===========================================================================
# Task 4 -- Fitness vs frequency+WAE: geometry vs AE, all-panel vs test-only, avg over test panels
# ===========================================================================
def plot_fitness_vs_freq_across_tests(out_dir=OUT_DIR):
    '''Fitness (ensemble fold) vs the 6 raw frequencies + WAE, one line each for CAE-GCN
    Fitness, its test Fitness, CAE Fitness, and its
    test Fitness -- every point averaged (nanmean) across the 5 test-panel runs.'''
    out_dir.mkdir(parents=True, exist_ok=True)
    n_freq = len(FREQUENCY_MAPPING)
    series = {key: np.full((len(ALL_PANELS), n_freq + 1), np.nan)
              for key in ("geometry", "geometry_test", "path", "path_test")}

    for t_idx, test_panel in enumerate(ALL_PANELS):
        for model_type, all_key, test_key in [("geometry", "geometry", "geometry_test"), ("path", "path", "path_test")]:
            model_dir = _model_dir(test_panel, model_type)

            HI_metrics = _load_pickle(model_dir / "HI_metrics.pkl")
            if HI_metrics is not None:
                series[all_key][t_idx, :n_freq] = HI_metrics[_last_fold_idx(HI_metrics), :n_freq, 0]
            WAE_metrics = _load_pickle(model_dir / "WAE_HI_metrics.pkl")
            if WAE_metrics is not None:
                series[all_key][t_idx, n_freq] = WAE_metrics[_last_fold_idx(WAE_metrics), 0]

            HI_test_metrics = _load_pickle(model_dir / "HI_test_metrics.pkl")
            if HI_test_metrics is not None:
                series[test_key][t_idx, :n_freq] = HI_test_metrics[_last_fold_idx(HI_test_metrics), :n_freq, 0]
            WAE_test_metrics = _load_pickle(model_dir / "WAE_HI_test_metrics.pkl")
            if WAE_test_metrics is not None:
                series[test_key][t_idx, n_freq] = WAE_test_metrics[_last_fold_idx(WAE_test_metrics), 0]

    x = np.arange(len(FREQ_LABELS))
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_labels = {"geometry": "GCN (geometry) Fitness", "geometry_test": "GCN (geometry) Fitness (test)",
                   "path": "Path AE ensemble Fitness", "path_test": "Path AE ensemble Fitness (test)"}
    with np.errstate(all="ignore"):
        for i, key in enumerate(["geometry", "geometry_test", "path", "path_test"]):
            is_test = key.endswith("_test")
            color, _ = _palette_style(i // 2)
            avg = np.nanmean(series[key], axis=0)
            ax.plot(x, avg, marker="s" if is_test else "o", linestyle="--" if is_test else "-",
                    color=color, label=plot_labels[key])
    ax.set_xticks(x)
    ax.set_xticklabels(FREQ_LABELS, rotation=45, ha="right")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Fitness ")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    save_path = out_dir / "Fitness_vs_frequency_GCN_AE_test_comparison.svg"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return series


# ===========================================================================
# Task 5 -- WAE Fitness/HI metrics vs GCN_TYPES, avg over test panels
# ===========================================================================
_STACK_METRICS = ["monotonicity", "prognosability", "trendability"]  # sum of the 3 = fitness
_STACK_LABELS = ["Mo", "Pr", "Tr"]

LOSS_ABLATION_TYPES = ["peak_tff", "peak_fft", "peak_tft"]

def plot_WAE_fitness_vs_types_across_tests(types=None, out_dir=OUT_DIR, file_tag=None):
    '''Grouped stacked-bar chart: one x-position per `types` entry (default MODEL_TYPES),
    two bars per type ("all panels" / "test panel only"), each bar stacking WAE Mo+Pr+Tr
    (ensemble fold, so bar height = WAE Fitness) -- every segment averaged across the 5
    test-panel runs. `file_tag` distinguishes the saved file when called more than once
    with different `types` subsets.'''
    types = types if types is not None else MODEL_TYPES
    out_dir.mkdir(parents=True, exist_ok=True)

    all_vals = {m: np.full(len(types), np.nan) for m in _STACK_METRICS}
    test_vals = {m: np.full(len(types), np.nan) for m in _STACK_METRICS}

    for ti, model_type in enumerate(types):
        for metric in _STACK_METRICS:
            metric_idx = METRIC_COLUMNS.index(metric)
            all_series, test_series = [], []
            for test_panel in ALL_PANELS:
                fold_idx = _ensemble_fold_idx(test_panel)
                model_dir = _model_dir(test_panel, model_type)

                WAE_metrics = _load_pickle(model_dir / "WAE_HI_metrics.pkl")
                if WAE_metrics is not None:
                    all_series.append(WAE_metrics[fold_idx, metric_idx])

                WAE_test_metrics = _load_pickle(model_dir / "WAE_HI_test_metrics.pkl")
                if WAE_test_metrics is not None:
                    test_series.append(WAE_test_metrics[fold_idx, metric_idx])

            if all_series:
                all_vals[metric][ti] = np.mean(all_series)
            if test_series:
                test_vals[metric][ti] = np.mean(test_series)

    x = np.arange(len(types))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom_all = np.zeros(len(types))
    bottom_test = np.zeros(len(types))
    for i, (metric, label) in enumerate(zip(_STACK_METRICS, _STACK_LABELS)):
        color = PALETTE[i % len(PALETTE)]
        seg_all = np.nan_to_num(all_vals[metric])
        seg_test = np.nan_to_num(test_vals[metric])
        ax.bar(x - width / 2, seg_all, width, bottom=bottom_all, color=color, label=label)
        ax.bar(x + width / 2, seg_test, width, bottom=bottom_test, color=color, hatch="//", edgecolor="white")
        bottom_all += seg_all
        bottom_test += seg_test

    metric_handles = [Patch(facecolor=PALETTE[i % len(PALETTE)], label=label) for i, label in enumerate(_STACK_LABELS)]
    style_handles = [Patch(facecolor="gray", label="All panels"), Patch(facecolor="gray", hatch="//", edgecolor="white", label="Test panel only")]
    all_handles = metric_handles + style_handles
    ax.legend(handles=all_handles, fontsize=8, ncol=len(all_handles), loc="upper center", bbox_to_anchor=(0.5, -0.2))

    ax.set_xticks(x)
    ax.set_xticklabels([TYPES_LABELS.get(t, t) for t in types], rotation=45, ha="right")
    ax.set_ylabel("WAE Fitness")
    ax.grid(True, axis="y")
    fig.tight_layout()
    suffix = f"_{file_tag}" if file_tag else ""
    save_path = out_dir / f"WAE_fitness_vs_GCN_types_across_tests{suffix}.svg"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return all_vals, test_vals


def main():
    
    save_HI_metrics_summary_xlsx()  # Task 2.1 + 2.2

    for model_type in MODEL_TYPES:  # for CAE, CAE-GCN and GCN      
        plot_HI_grid_across_tests(model_type)  # Task 1

        plot_damage_map_grid_across_tests(model_type, normalize=False)  # Task 3
       
    plot_fitness_vs_freq_across_tests()  # Task 4
    
    plot_WAE_fitness_vs_types_across_tests(types=MODEL_TYPES, file_tag="adjacency")  # Task 5
    
    # if loss sensitivity study is done 
    # plot_WAE_fitness_vs_types_across_tests(types=["peak", "raw", "path"] + LOSS_ABLATION_TYPES, file_tag="loss")  


if __name__ == "__main__":
    main()
