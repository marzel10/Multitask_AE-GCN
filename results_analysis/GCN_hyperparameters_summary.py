'''
Summarizes the pre-processed-features GCN's Bayesian-optimized hyperparameters across
every test panel, read from
TEST_RUN_DIR/"results"/f"best_hyperparameters_freq{freq}{beta_suffix}_pre_processed.json"
across all 4 test_{panel}_wo123 runs (BO_GCN.py's run_bayesian_optimization, raw_features=False,
type="peak").

Plots, one per hyperparameter, frequency index on the x-axis, one colored series per test
panel (up to 4 points per frequency):
    - nr_hidden_channels
    - hidden_dim
    - dropout
    - batch_size
    - learning_rate

Panel/frequency combos whose json doesn't exist yet are skipped rather than erroring.
'''
import json
import sys
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

from config import PROJECT_ROOT, CUSTOM_PALETTE, PANEL_LABELS

TEST_PANELS = ["103", "104", "105", "109"]
FREQS = range(0, 6)
BETA_SUFFIX = ""
OUT_DIR = PROJECT_ROOT / "GCN_hyperparameters_summary_results"

_MARKERS = ["o", "s", "^", "D"]


def _palette_style(i):
    color = CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)]
    marker = _MARKERS[(i // len(CUSTOM_PALETTE)) % len(_MARKERS)]
    return color, marker


HP_PLOTS = [
    ("nr_hidden_channels", "Optimal nr_hidden_channels"),
    ("hidden_dim", "Optimal hidden_dim"),
    ("dropout", "Optimal dropout"),
    ("batch_size", "Optimal batch_size"),
    ("learning_rate", "Optimal learning_rate"),
]


def load_best_params(panels=TEST_PANELS, freqs=FREQS, beta_suffix=BETA_SUFFIX, raw=False):
    rows = []
    for panel in panels:
        root = PROJECT_ROOT / f"test_{panel}_wo123"
        for freq in freqs:
            if raw:
                json_path = root / "results" / f"best_hyperparameters_freq{freq}{beta_suffix}.json"
            else:
                json_path = root / "results" / f"best_hyperparameters_freq{freq}{beta_suffix}_pre_processed.json"
            if not json_path.exists():
                print(f"[panel {panel}, freq {freq}] no best_hyperparameters json, skipping")
                continue
            with open(json_path) as f:
                params = json.load(f)
            row = {"panel": panel, "frequency_index": freq, **params}
            rows.append(row)
            print(f"[panel {panel}, freq {freq}] loaded best_hyperparameters json")

    if not rows:
        raise FileNotFoundError(f"No best_hyperparameters_*_pre_processed.json found for any of {panels} x {list(freqs)}")

    return pd.DataFrame(rows)


def plot_hp_vs_frequency(df, out_dir=OUT_DIR, raw=False):
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [p for p in TEST_PANELS if p in set(df["panel"])]

    for col, title in HP_PLOTS:
        if col not in df.columns:
            print(f"[{col}] column missing from every loaded best_hyperparameters json, skipping plot")
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
        ax.legend(title="Test panel", fontsize=14, title_fontsize=15, loc="best")
        ax.grid(True)
        fig.tight_layout()
        if raw:
            save_path = out_dir / f"GCN_hp_summary_{col}_raw.svg"
        else:
            save_path = out_dir / f"GCN_hp_summary_{col}.svg"
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved: {save_path}")



def save_hp_summary_xlsx(df, out_dir=OUT_DIR, raw=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    if raw:
        save_path = out_dir / "GCN_hp_summary_raw.xlsx"
    else:
        save_path = out_dir / "GCN_hp_summary.xlsx"
    with pd.ExcelWriter(save_path) as writer:
        for col, title in HP_PLOTS:
            if col not in df.columns:
                print(f"[{col}] column missing from every loaded best_hyperparameters json, skipping sheet")
                continue
            sub = df.dropna(subset=[col, "frequency_index", "panel"])
            if sub.empty:
                print(f"[{col}] no non-null rows, skipping sheet")
                continue
            sub.to_excel(writer, sheet_name=col, index=False)
    print(f"Saved: {save_path}")


def main():
    df = load_best_params()
    df_raw = load_best_params(raw=True)
    print(f"Loaded {len(df)} row(s) total across {df['panel'].nunique()} test panel(s)")
    print(f"Loaded {len(df_raw)} row(s) total across {df_raw['panel'].nunique()} test panel(s)")
    plot_hp_vs_frequency(df)
    save_hp_summary_xlsx(df)
    plot_hp_vs_frequency(df_raw, raw=True)
    save_hp_summary_xlsx(df_raw, raw=True)


if __name__ == "__main__":
    main()
