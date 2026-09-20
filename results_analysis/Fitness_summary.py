'''
Cross-model-type summary of the (frequency x path x metric) `metrics.npy` arrays
produced by graph_performance.py (graph_performance_results_<type>/, one per GCN
adjacency-matrix type in training/sweep_over_options.py's `types` + the raw-feature
variant) and path_performance.py (path_performance_results/, the per-path AE
ensemble).

Every metrics.npy shares the same layout regardless of which script wrote it:
frequency (6 raw + 1 average over frequencies) x path (28) x metric (Fitness, Mo,
Pr, Tr). This module averages across paths (nanmean, so a handful of missing/NaN
paths don't blank a whole row -- relevant for path_performance_results, whose BO
runs may still be incomplete for some paths) to get one (model_type, frequency) ->
(fitness, monotonicity, prognosability, trendability) table, writes it to an
.xlsx, and plots each metric vs frequency, one line per model type.
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import TEST_RUN_DIR, GRAPH_TYPES, GRAPH_LABELS, FREQ_LABELS, FOLD_KEYS, METRIC_COLUMNS, METRIC_NAMES, OUT_XLSX, OUT_DIR, CUSTOM_PALETTE as PALETTE, _LINESTYLES


def _palette_style(i):
    color = PALETTE[i % len(PALETTE)]
    linestyle = _LINESTYLES[(i // len(PALETTE)) % len(_LINESTYLES)]
    return color, linestyle


def _model_dirs():
    dirs = {t: TEST_RUN_DIR / f"graph_performance_results_{t}" for t in GRAPH_TYPES}
    dirs["path"] = TEST_RUN_DIR / "path_performance_results"
    dirs["raw"] = TEST_RUN_DIR / "graph_performance_results_raw"
    return dirs


def load_summary(model_dirs=None, filename="HI_metrics.pkl"):
    '''Load `filename` from each model_dir -- ensemble fold only, one row per frequency.
    `filename` defaults to HI_metrics.pkl (overall metrics); pass "HI_test_metrics.pkl"
    to load the test-panel-only metrics produced by test_metrics.py instead.'''
    model_dirs = model_dirs or _model_dirs()
    wae_filename = f"WAE_{filename}"
    rows = []
    for model_type, folder in model_dirs.items():
        metrics_path = folder / filename
        if not metrics_path.exists():
            print(f"[{model_type}] no {filename} in {folder}, skipping")
            continue
        metrics = np.load(metrics_path, allow_pickle=True)  # (folds, freq, metric)
        ensemble_stats = metrics[-1].copy() #(freq, metric)

        wae_path = folder / wae_filename
        if wae_path.exists():
            wae_metrics = np.load(wae_path, allow_pickle=True)  # (folds, metric)
            ensemble_stats[-1] = wae_metrics[-1]  # replace plain average with the WAE combination
        else:
            print(f"[{model_type}] no {wae_filename} in {folder}, 'average' column stays a plain average")

        for freq_idx, freq_label in enumerate(FREQ_LABELS):
            row = {"model_type": model_type, "frequency": freq_label}
            row.update(zip(METRIC_COLUMNS, ensemble_stats[freq_idx]))
            rows.append(row)
        print(f"[{model_type}] loaded {metrics_path}")

    if not rows:
        raise FileNotFoundError(
            f"No {filename} found for any model type (looked under {list(model_dirs.values())})"
        )

    return pd.DataFrame(rows)


def load_wae_summary(model_dirs=None, filename="WAE_HI_metrics.pkl"):
    '''Load `filename` (a (fold, metric) WAE_HI_metrics.pkl produced by Compute_WAE.py)'''
    model_dirs = model_dirs or _model_dirs()
    rows = []
    for model_type, folder in model_dirs.items():
        metrics_path = folder / filename
        if not metrics_path.exists():
            print(f"[{model_type}] no {filename} in {folder}, skipping")
            continue
        metrics = np.load(metrics_path, allow_pickle=True)  # (folds, metric)

        for fold_label, fold_stats in zip(FOLD_KEYS, metrics):
            row = {"model_type": model_type, "fold": fold_label}
            row.update(zip(METRIC_COLUMNS, fold_stats))
            rows.append(row)
        print(f"[{model_type}] loaded {metrics_path}")

    if not rows:
        raise FileNotFoundError(
            f"No {filename} found for any model type (looked under {list(model_dirs.values())})"
        )

    return pd.DataFrame(rows)


def save_summary(df, out_path=OUT_XLSX, sheet_name="metrics"):
    '''Writes `df` as one sheet of out_path, without clobbering the workbook's other sheets.'''
    mode = "a" if out_path.exists() else "w"
    kwargs = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(out_path, engine="openpyxl", mode=mode, **kwargs) as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    print(f"Saved: {out_path} [{sheet_name}]")


def plot_summary(df, out_dir=OUT_DIR, model_dirs=None, GCN_AE_comp=False, raw_comp=False, test_df=None):
    '''One figure per metric: metric value vs frequency, one line per model type. 
    If test data is available, it will be plotted as a second line for each model type.
    '''
    out_dir.mkdir(parents=True, exist_ok=True)
    model_types = list((model_dirs or _model_dirs()).keys())
    x = np.arange(len(FREQ_LABELS))

    for metric_col, metric_name in zip(METRIC_COLUMNS, METRIC_NAMES):
        fig, ax = plt.subplots(figsize=(9, 5))
        any_series = False
        for i, model_type in enumerate(model_types):

            if GCN_AE_comp:
                if model_type != "path" and model_type != "peak":
                    continue  # only plot the "peak" GCN (baseline adjacency type) and the path AE ensemble
                label = "Path AE ensemble" if model_type == "path" else "GCN"
            elif raw_comp:
                if model_type != "peak" and model_type != "raw":
                    continue  # only plot the raw-feature GCN and the baseline "peak" GCN
                label = "Raw features GCN" if model_type == "raw" else "GCN"
            else:
                if model_type == "path" or model_type == "raw":
                    continue  # only plot the GCN adjacency-matrix types
                label = GRAPH_LABELS.get(model_type, model_type)
            sub = df[df["model_type"] == model_type]
            if sub.empty:
                continue
            sub = sub.set_index("frequency").reindex(FREQ_LABELS)
            color, linestyle = _palette_style(i)
            ax.plot(x, sub[metric_col].values, marker="o", linestyle=linestyle, color=color, label=label)
            any_series = True

            if GCN_AE_comp and test_df is not None:
                test_sub = test_df[test_df["model_type"] == model_type]
                if not test_sub.empty:
                    test_sub = test_sub.set_index("frequency").reindex(FREQ_LABELS)
                    ax.plot(x, test_sub[metric_col].values, marker="s", linestyle="--",
                            color=color, alpha=0.6, label=f"{label} (test)")
        if not any_series:
            plt.close(fig)
            continue
        ax.set_xticks(x)
        ax.set_xticklabels(FREQ_LABELS, rotation=45, ha="right")
        ax.set_xlabel("Frequency")
        ax.set_ylabel(metric_name)
        #ax.set_title(f"{metric_name} vs Frequency, by model type (averaged over paths)")
        if GCN_AE_comp or raw_comp:
            legend_title = "Model type"
        else:
            legend_title = "Adjacency matrix type"
        ax.legend(title=legend_title, handlelength=3, fontsize=8)
        ax.grid(True)
        fig.tight_layout()
        if GCN_AE_comp:
            save_path = out_dir / f"{metric_name}_vs_frequency_GCN_AE_comp.svg"
        elif raw_comp:
            save_path = out_dir / f"{metric_name}_vs_frequency_raw_comp.svg"
        else:
            save_path = out_dir / f"{metric_name}_vs_frequency.svg"
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved: {save_path}")


def main():
    df = load_summary()

    print(f"Loaded {len(df)} row(s) across {df['model_type'].nunique()} model type(s)")
    save_summary(df)

    try:
        test_df = load_summary(filename="HI_test_metrics.pkl")
        print(f"Loaded {len(test_df)} test-metric row(s) across {test_df['model_type'].nunique()} model type(s)")
        save_summary(test_df, sheet_name="test_metrics")
    except FileNotFoundError as e:
        print(f"No HI_test_metrics.pkl found, skipping test-metric overlay: {e}")
        test_df = None

    try:
        wae_df = load_wae_summary()
        print(f"Loaded {len(wae_df)} WAE-metric row(s) across {wae_df['model_type'].nunique()} model type(s)")
        save_summary(wae_df, sheet_name="WAE_metrics")
    except FileNotFoundError as e:
        print(f"No WAE_HI_metrics.pkl found, skipping WAE sheet: {e}")

    try:
        wae_test_df = load_wae_summary(filename="WAE_HI_test_metrics.pkl")
        print(f"Loaded {len(wae_test_df)} WAE test-metric row(s) across {wae_test_df['model_type'].nunique()} model type(s)")
        save_summary(wae_test_df, sheet_name="WAE_test_metrics")
    except FileNotFoundError as e:
        print(f"No WAE_HI_test_metrics.pkl found, skipping WAE test sheet: {e}")

    plot_summary(df)
    plot_summary(df, GCN_AE_comp=True, test_df=test_df)
    plot_summary(df, raw_comp=True)


if __name__ == "__main__":
    main()
