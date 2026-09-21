'''
Summarizes the Bayesian-optimized CNN_AE hyperparameters across every test panel, read
from the "best_params.json" files that BO_AE.py's run_bayesian_optimization writes once
per Multi_path_BO_fixed_freq{N} folder -- these come from the Bayesian search run against
path 0, and are the hyperparameters shared by every path within that TEST_PANEL/frequency
combo (see BO_AE.py's params_path).

Plots, one per hyperparameter, frequency index on the x-axis, one colored series per test
panel (4 points per frequency -- one per panel's independent Bayesian search):
    - k_sparse   (the resolved absolute count -- see _resolve_k_sparse below -- not the
                  k_sparse_frac hyperparameter the search actually samples)
    - filters_bench
    - filters_path
    - batch_size

Panel/frequency combos whose best_params.json doesn't exist yet are skipped rather than
erroring.
'''
import sys
import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "tools", "training", "intermediate_results_check", "results_analysis"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

from config import PROJECT_ROOT, CNN_FIXED_LATENT_DIM, CUSTOM_PALETTE, PANEL_LABELS
TEST_PANELS = ["103", "104", "105", "109"]  # each has its own test_{panel}_wo123 run (that panel held out as TEST_PANEL)
FREQS = range(0, 6)
OUT_DIR = PROJECT_ROOT / "AE_hyperparameters_summary_results"

_MARKERS = ["o", "s", "^", "D"]


def _palette_style(i):
    color = CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)]
    marker = _MARKERS[(i // len(CUSTOM_PALETTE)) % len(_MARKERS)]
    return color, marker


def _resolve_k_sparse(k_sparse_frac, latent_dim=CNN_FIXED_LATENT_DIM):
    '''Mirrors training.BO_AE.resolve_k_sparse -- reimplemented locally so this script
    doesn't have to import BO_AE (which pulls in TensorFlow/keras_tuner) for one line.'''
    return max(2, min(latent_dim - 1, round(k_sparse_frac * latent_dim)))


# (key in best_params.json / computed above, plot title / y-axis label)
HP_PLOTS = [
    ("k_sparse", "Optimal latent size (k-sparse)"),
    ("filters_bench", "Optimal benchmark filter count"),
    ("filters_path", "Optimal path filter count"),
    ("batch_size", "Optimal batch size"),
]


def load_best_params(panels=TEST_PANELS, freqs=FREQS):
    '''Loads path 0's best_params.json for every (test panel, frequency) combo into one
    DataFrame, tagging each row with its panel and frequency_index.'''
    rows = []
    for panel in panels:
        root = PROJECT_ROOT / f"test_{panel}_wo123"
        for freq in freqs:
            json_path = (root / f"Multi_path_BO_fixed_freq{freq}"
                         / f"Bayesian_CNN_AE_test{[panel]}_freq{freq}_best_params.json")
            if not json_path.exists():
                print(f"[panel {panel}, freq {freq}] no best_params.json, skipping")
                continue
            with open(json_path) as f:
                params = json.load(f)
            row = {"panel": panel, "frequency_index": freq, **params}
            if "k_sparse_frac" in params:
                row["k_sparse"] = _resolve_k_sparse(params["k_sparse_frac"])
            rows.append(row)
            print(f"[panel {panel}, freq {freq}] loaded best_params.json")

    if not rows:
        raise FileNotFoundError(f"No best_params.json found for any of {panels} x {list(freqs)}")

    return pd.DataFrame(rows)


def plot_hp_vs_frequency(df, out_dir=OUT_DIR):
    '''One figure per hyperparameter: value vs frequency_index, one series per test panel
    (colors = panel, so each frequency shows 4 points -- one per test panel's own
    Bayesian search).'''
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [p for p in TEST_PANELS if p in set(df["panel"])]

    for col, title in HP_PLOTS:
        if col not in df.columns:
            print(f"[{col}] column missing from every loaded best_params.json, skipping plot")
            continue
        sub = df.dropna(subset=[col, "frequency_index", "panel"])
        if sub.empty:
            print(f"[{col}] no non-null rows, skipping plot")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        any_series = False
        for i, panel in enumerate(panels):
            panel_rows = sub[sub["panel"] == panel].sort_values("frequency_index")
            if panel_rows.empty:
                continue
            color, marker = _palette_style(i)
            ax.plot(panel_rows["frequency_index"], panel_rows[col], marker=marker, linestyle="None",
                    color=color, label=f"{PANEL_LABELS.get(panel, f'panel {panel}')}")
            any_series = True
        if not any_series:
            plt.close(fig)
            continue

        ax.set_xlabel("Frequency index", fontsize=15)
        ax.set_ylabel(title, fontsize=15)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(title="Test panel", fontsize=14, title_fontsize=15, loc="best")
        ax.grid(True)
        fig.tight_layout()
        save_path = out_dir / f"AE_hp_summary_{col}.svg"
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved: {save_path}")

def save_hp_summary_xlsx(df, out_dir=OUT_DIR):
    
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "AE_hp_summary.xlsx"
    with pd.ExcelWriter(save_path) as writer:
        for col, title in HP_PLOTS:
            if col not in df.columns:
                print(f"[{col}] column missing from every loaded best_params.json, skipping sheet")
                continue
            sub = df.dropna(subset=[col, "frequency_index", "panel"])
            if sub.empty:
                print(f"[{col}] no non-null rows, skipping sheet")
                continue
            sub.to_excel(writer, sheet_name=col, index=False)
    print(f"Saved: {save_path}")

def main():
    df = load_best_params()
    print(f"Loaded {len(df)} row(s) total across {df['panel'].nunique()} test panel(s)")
    plot_hp_vs_frequency(df)
    save_hp_summary_xlsx(df)

if __name__ == "__main__":
    main()
