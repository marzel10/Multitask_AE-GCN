'''
Diagnostic: for every panel's cached AE-latent raw file(s)
(panel_<panel>_shi_raw_<freq>[_fold<fold>].pt, produced by extract_shi.py under
GRAPH_DATA_DIR/raw/), reports per big_latent vector position:
  - how many (path, state) entries are exactly zero (KSparse hard-zeros all but the
    top-k activations, see models/CNN_AE.py)
  - the [min, max] value range that position actually takes
  - the mean and std of the values it takes
across every cached frequency/fold found for that panel -- a quick sanity check for
how sparse/how wide-ranging the AE's K-Sparse latent code is per dimension.
'''
import re
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
import torch

from config import GRAPH_DATA_DIR

_RAW_NAME_RE = re.compile(r"^panel_(.+)_shi_raw_\d+(?:_fold.+)?\.pt$")
_RAW_FILE_RE = re.compile(r"^panel_(.+)_shi_raw_(\d+)(?:_fold(.+))?\.pt$")


def latent_stats_per_position(raw_dir=None):
    raw_dir = raw_dir or (GRAPH_DATA_DIR / "raw")
    zero_counts, n_entries, mins, maxs, sums, sumsqs = {}, {}, {}, {}, {}, {}

    for path in sorted(raw_dir.glob("panel_*_shi_raw_*.pt")):
        m = _RAW_NAME_RE.match(path.name)
        if not m:
            continue
        panel = m.group(1)

        raw_data = torch.load(path, weights_only=False)
        big_latent = np.asarray(raw_data["big_latent"])  # paths x states x latent_dim
        flat = big_latent.reshape(-1, big_latent.shape[-1]).astype(np.float64)  # (paths * states, latent_dim)

        counts = (flat == 0).sum(axis=0)
        file_min = flat.min(axis=0)
        file_max = flat.max(axis=0)
        file_sum = flat.sum(axis=0)
        file_sumsq = (flat ** 2).sum(axis=0)

        if panel not in zero_counts:
            zero_counts[panel] = counts
            n_entries[panel] = flat.shape[0]
            mins[panel] = file_min
            maxs[panel] = file_max
            sums[panel] = file_sum
            sumsqs[panel] = file_sumsq
        else:
            zero_counts[panel] = zero_counts[panel] + counts
            n_entries[panel] = n_entries[panel] + flat.shape[0]
            mins[panel] = np.minimum(mins[panel], file_min)
            maxs[panel] = np.maximum(maxs[panel], file_max)
            sums[panel] = sums[panel] + file_sum
            sumsqs[panel] = sumsqs[panel] + file_sumsq

    return zero_counts, n_entries, mins, maxs, sums, sumsqs


def find_extreme_entries(raw_dir=None, top_n=30):
    '''Across every panel_*_shi_raw_*.pt file, the top_n (panel, freq, fold, path, state,
    position) entries with the largest |value| in big_latent -- to see whether the extreme
    magnitudes'''
    raw_dir = raw_dir or (GRAPH_DATA_DIR / "raw")
    records = []  # (abs_value, value, panel, freq, fold, path, state, position)

    for path in sorted(raw_dir.glob("panel_*_shi_raw_*.pt")):
        m = _RAW_FILE_RE.match(path.name)
        if not m:
            continue
        panel, freq, fold = m.group(1), m.group(2), m.group(3) or "ensemble"

        raw_data = torch.load(path, weights_only=False)
        big_latent = np.asarray(raw_data["big_latent"])  # paths x states x latent_dim

        # top_n candidates from this file alone are enough -- the file-level top_n can
        # only shrink once merged with every other file's candidates below.
        flat = np.abs(big_latent).ravel()
        n_candidates = min(top_n, flat.size)
        top_flat_idx = np.argpartition(flat, -n_candidates)[-n_candidates:]
        path_idx, state_idx, pos_idx = np.unravel_index(top_flat_idx, big_latent.shape)
        for p_i, s_i, pos_i in zip(path_idx, state_idx, pos_idx):
            val = float(big_latent[p_i, s_i, pos_i])
            records.append((abs(val), val, panel, freq, fold, int(p_i), int(s_i), int(pos_i)))

    records.sort(key=lambda r: r[0], reverse=True)
    return records[:top_n]


def main(raw_dir=None):
    zero_counts, n_entries, mins, maxs, sums, sumsqs = latent_stats_per_position(raw_dir)
    if not zero_counts:
        print(f"No panel_*_shi_raw_*.pt files found in {raw_dir or (GRAPH_DATA_DIR / 'raw')}")
        return

    for panel in sorted(zero_counts, key=lambda p: (len(p), p)):
        counts = zero_counts[panel]
        n = n_entries[panel]
        mean = sums[panel] / n
        std = np.sqrt(np.maximum(sumsqs[panel] / n - mean ** 2, 0.0))
        print(f"\nPanel {panel} (n={n} path*state entries per position):")
        for pos, c in enumerate(counts):
            print(f"  pos {pos:2d}: zeroed {c:6d}/{n} times ({c / n:.1%})  "
                  f"range=[{mins[panel][pos]:.4f}, {maxs[panel][pos]:.4f}]  "
                  f"mean={mean[pos]:.4f}  std={std[pos]:.4f}")

    top_n = 30
    extremes = find_extreme_entries(raw_dir, top_n=top_n)
    print(f"\nTop {len(extremes)} most extreme |value| entries (panel, freq, fold, path, state, position, value):")
    for abs_val, val, panel, freq, fold, p_i, s_i, pos_i in extremes:
        print(f"  panel={panel:>5} freq={freq} fold={fold:>4}  path={p_i:2d}  state={s_i:3d}  pos={pos_i:2d}  value={val:.4e}")

    from collections import Counter
    path_counts = Counter(p_i for _, _, _, _, _, p_i, _, _ in extremes)
    state_counts = Counter((panel, s_i) for _, _, panel, _, _, _, s_i, _ in extremes)
    print(f"\nAmong those {len(extremes)} entries, path index frequency: {dict(sorted(path_counts.items()))}")
    print(f"(panel, state) frequency: {dict(sorted(state_counts.items()))}")


if __name__ == "__main__":
    main()
