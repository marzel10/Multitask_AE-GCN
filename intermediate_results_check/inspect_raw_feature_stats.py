'''
Produces similar stats as intermediate_results_check/inspect_latent_zeros.py,
but for the raw features instead of the AE's latent space.

Comparison between the two stats is made for verification wheter the latent space extraction works correctly.
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

from config import GRAPH_DATA_DIR

_RAW_FEATURE_RE = re.compile(r"^States_(.+)_freq(\d+)_all_paths_diff_full\.npy$")


def raw_feature_stats(raw_dir=None):
    '''{(panel, freq): (mean, std, n)} -- mean/std are length-n_features arrays over
    every (path, state) entry in that panel/freq's cached raw-feature file, computed in
    float64 for precision; n is the number of (path, state) entries tallied.'''
    raw_dir = raw_dir or (GRAPH_DATA_DIR / "raw")
    stats = {}

    for path in sorted(raw_dir.glob("States_*_all_paths_diff_full.npy")):
        m = _RAW_FEATURE_RE.match(path.name)
        if not m:
            continue
        panel, freq = m.group(1), m.group(2)

        features = np.load(path).astype(np.float64)  # paths x states x n_features
        flat = features.reshape(-1, features.shape[-1])  # (paths * states, n_features)

        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        stats[(panel, freq)] = (mean, std, flat.shape[0])

    return stats


def main(raw_dir=None):
    stats = raw_feature_stats(raw_dir)
    if not stats:
        print(f"No States_*_all_paths_diff_full.npy files found in {raw_dir or (GRAPH_DATA_DIR / 'raw')}")
        return

    for panel, freq in sorted(stats, key=lambda k: (len(k[0]), k[0], int(k[1]))):
        mean, std, n = stats[(panel, freq)]
        print(f"\nPanel {panel}, freq {freq} (n={n} path*state entries per position):")
        for pos in range(len(mean)):
            print(f"  pos {pos:2d}: mean={mean[pos]:.4f}  std={std[pos]:.4f}")
        input("Press Enter to continue to next panel/freq...")

if __name__ == "__main__":
    main()
