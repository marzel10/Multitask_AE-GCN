'''
Same accumulation approach as inspect_latent_zeros.py's latent_stats_per_position, just
keyed by path/state instead of latent position, and on raw_data["shi"] (paths x states x
1) instead of raw_data["big_latent"].

Used for further validation of the latent space extraction 
'''
import re
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "tools", "training", "intermediate_results_check", "results_analysis"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from config import GRAPH_DATA_DIR

_RAW_NAME_RE = re.compile(r"^panel_(.+)_shi_raw_\d+(?:_fold.+)?\.pt$")
_RAW_FILE_RE = re.compile(r"^panel_(.+)_shi_raw_(\d+)(?:_fold(.+))?\.pt$")


def shi_stats_per_path(raw_dir=None):
    '''{panel: (n, mins, maxs, sums, sumsqs)} -- mins/maxs/sums/sumsqs are length-28
    arrays (one entry per path), n is the number of states tallied per path; all summed/
    reduced across every panel_<panel>_shi_raw_*.pt file found in raw_dir (i.e. every
    cached frequency/fold for that panel). Accumulated in float64 since sHI can explode
    to extreme magnitude for some (panel, freq, fold) combos (see performance_accross_
    tests.py's _normalize_maps) and would lose precision or overflow in float32.'''
    raw_dir = raw_dir or (GRAPH_DATA_DIR / "raw")
    n_entries, mins, maxs, sums, sumsqs = {}, {}, {}, {}, {}

    for path in sorted(raw_dir.glob("panel_*_shi_raw_*.pt")):
        m = _RAW_NAME_RE.match(path.name)
        if not m:
            continue
        panel = m.group(1)

        raw_data = torch.load(path, weights_only=False)
        shi = np.asarray(raw_data["shi"]).astype(np.float64)  # paths x states x 1
        shi = shi.reshape(shi.shape[0], shi.shape[1])  # paths x states

        file_min = shi.min(axis=1)
        file_max = shi.max(axis=1)
        file_sum = shi.sum(axis=1)
        file_sumsq = (shi ** 2).sum(axis=1)

        if panel not in n_entries:
            n_entries[panel] = shi.shape[1]
            mins[panel] = file_min
            maxs[panel] = file_max
            sums[panel] = file_sum
            sumsqs[panel] = file_sumsq
        else:
            n_entries[panel] = n_entries[panel] + shi.shape[1]
            mins[panel] = np.minimum(mins[panel], file_min)
            maxs[panel] = np.maximum(maxs[panel], file_max)
            sums[panel] = sums[panel] + file_sum
            sumsqs[panel] = sumsqs[panel] + file_sumsq

    return n_entries, mins, maxs, sums, sumsqs


def shi_stats_per_state(raw_dir=None):
    '''{panel: (n, mins, maxs, sums, sumsqs)} -- the transpose of shi_stats_per_path:
    mins/maxs/sums/sumsqs are length-n_states arrays (one entry per state), n is the
    number of paths tallied per state; summed/reduced across every
    panel_<panel>_shi_raw_*.pt file found in raw_dir. A file whose state count doesn't
    match what's already accumulated for that panel (e.g. a stale/differently-shaped
    cache) is skipped with a warning rather than raising.'''
    raw_dir = raw_dir or (GRAPH_DATA_DIR / "raw")
    n_entries, mins, maxs, sums, sumsqs = {}, {}, {}, {}, {}

    for path in sorted(raw_dir.glob("panel_*_shi_raw_*.pt")):
        print(f"Processing {path.name}")
        input("Press Enter to continue...")
        m = _RAW_NAME_RE.match(path.name)
        if not m:
            continue
        panel = m.group(1)

        raw_data = torch.load(path, weights_only=False)
        shi = np.asarray(raw_data["shi"]).astype(np.float64)  # paths x states x 1
        print(shi)
        mean = np.mean(shi, axis=0)
        print(mean)
        print(mean.shape)
        input("Press Enter to continue...")
        shi = shi.reshape(shi.shape[0], shi.shape[1])  # paths x states

        file_min = shi.min(axis=0)
        file_max = shi.max(axis=0)
        file_sum = shi.sum(axis=0)
        file_sumsq = (shi ** 2).sum(axis=0)

        if panel not in n_entries:
            n_entries[panel] = shi.shape[0]
            mins[panel] = file_min
            maxs[panel] = file_max
            sums[panel] = file_sum
            sumsqs[panel] = file_sumsq
        else:
            if file_sum.shape != sums[panel].shape:
                print(f"  [{path.name}] state count {file_sum.shape[0]} != {sums[panel].shape[0]} already accumulated for panel {panel}, skipping for per-state stats")
                continue
            n_entries[panel] = n_entries[panel] + shi.shape[0]
            mins[panel] = np.minimum(mins[panel], file_min)
            maxs[panel] = np.maximum(maxs[panel], file_max)
            sums[panel] = sums[panel] + file_sum
            sumsqs[panel] = sumsqs[panel] + file_sumsq

    return n_entries, mins, maxs, sums, sumsqs


def find_extreme_shi_entries(raw_dir=None, top_n=30):
    '''Across every panel_*_shi_raw_*.pt file, the top_n (panel, freq, fold, path, state)
    entries with the largest |sHI| -- to see whether extreme magnitudes are concentrated
    in a handful of specific paths, or spread evenly across the dataset.'''
    raw_dir = raw_dir or (GRAPH_DATA_DIR / "raw")
    records = []  # (abs_value, value, panel, freq, fold, path, state)

    for path in sorted(raw_dir.glob("panel_*_shi_raw_*.pt")):
        m = _RAW_FILE_RE.match(path.name)
        if not m:
            continue
        panel, freq, fold = m.group(1), m.group(2), m.group(3) or "ensemble"

        raw_data = torch.load(path, weights_only=False)
        shi = np.asarray(raw_data["shi"])  # paths x states x 1
        shi = shi.reshape(shi.shape[0], shi.shape[1])  # paths x states

        flat = np.abs(shi).ravel()
        n_candidates = min(top_n, flat.size)
        top_flat_idx = np.argpartition(flat, -n_candidates)[-n_candidates:]
        path_idx, state_idx = np.unravel_index(top_flat_idx, shi.shape)
        for p_i, s_i in zip(path_idx, state_idx):
            val = float(shi[p_i, s_i])
            records.append((abs(val), val, panel, freq, fold, int(p_i), int(s_i)))

    records.sort(key=lambda r: r[0], reverse=True)
    return records[:top_n]


def main(raw_dir=None):
    n_entries, mins, maxs, sums, sumsqs = shi_stats_per_path(raw_dir)
    if not n_entries:
        print(f"No panel_*_shi_raw_*.pt files found in {raw_dir or (GRAPH_DATA_DIR / 'raw')}")
        return

    for panel in sorted(n_entries, key=lambda p: (len(p), p)):
        n = n_entries[panel]
        mean = sums[panel] / n
        std = np.sqrt(np.maximum(sumsqs[panel] / n - mean ** 2, 0.0))
        print(f"\nPanel {panel} (n={n} states per path, aggregated over every cached freq/fold):")
        for path_i in range(len(mean)):
            print(f"  path {path_i:2d}: range=[{mins[panel][path_i]:.4f}, {maxs[panel][path_i]:.4f}]  "
                  f"mean={mean[path_i]:.4f}  std={std[path_i]:.4f}")

    n_entries_s, mins_s, maxs_s, sums_s, sumsqs_s = shi_stats_per_state(raw_dir)
    for panel in sorted(n_entries_s, key=lambda p: (len(p), p)):
        n = n_entries_s[panel]
        mean = sums_s[panel] / n
        std = np.sqrt(np.maximum(sumsqs_s[panel] / n - mean ** 2, 0.0))
        print(f"\nPanel {panel} (n={n} paths per state, aggregated over every cached freq/fold):")
        for state_i in range(len(mean)):
            print(f"  state {state_i:3d}: range=[{mins_s[panel][state_i]:.4f}, {maxs_s[panel][state_i]:.4f}]  "
                  f"mean={mean[state_i]:.4f}  std={std[state_i]:.4f}")

    top_n = 30
    extremes = find_extreme_shi_entries(raw_dir, top_n=top_n)
    print(f"\nTop {len(extremes)} most extreme |sHI| entries (panel, freq, fold, path, state, value):")
    for abs_val, val, panel, freq, fold, p_i, s_i in extremes:
        print(f"  panel={panel:>5} freq={freq} fold={fold:>4}  path={p_i:2d}  state={s_i:3d}  value={val:.4e}")

    path_counts = Counter(p_i for _, _, _, _, _, p_i, _ in extremes)
    state_counts = Counter((panel, s_i) for _, _, panel, _, _, _, s_i in extremes)
    print(f"\nAmong those {len(extremes)} entries, path index frequency: {dict(sorted(path_counts.items()))}")
    print(f"(panel, state) frequency: {dict(sorted(state_counts.items()))}")


if __name__ == "__main__":
    main()
