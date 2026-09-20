'''
Plots the "ellipse of influence" for one or more sensor paths, reusing
imagining_alghoritm.U's own R/beta weighting: a path's weight W is > 0 exactly
inside the ellipse with foci at its two sensors (R = (d1+d2)/d - 1 < beta defines it,
see U / optimal_beta in imagining_alghoritm.py) -- so the W > 0 boundary IS that ellipse.
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
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from imagining_alghoritm import U
from plot_panel import _draw_static_panel
from config import DEFAULT_N_PIXELS, PANEL_W, PANEL_H, SENSOR_POSITIONS, SENSOR_PAIRS, CUSTOM_PALETTE


def _build_grid(n_pixels=DEFAULT_N_PIXELS):
    dA = (PANEL_W * PANEL_H) / n_pixels
    dx = np.sqrt(dA)
    x = np.arange(0, PANEL_W + dx, dx)
    y = np.arange(0, PANEL_H + dx, dx)
    return np.meshgrid(x, y, indexing="ij")


def plot_ellipses_of_influence(paths, beta, n_pixels=DEFAULT_N_PIXELS,
                                panel_number=None, state=None, save_path=None):
    """
    Draws the ellipse of influence (U's W > 0 region) for each path in `paths`, on the panel.

    paths        : list of path numbers, 1-indexed (1-28), same convention as
                   visualize_crossing_graph.py.
    beta         : ellipse "width" parameter (see imagining_alghoritm.U / optimal_beta) --
                   defaults to None (per-path optimal_beta).
    panel_number, state : passed straight through to U (sensor-failure handling); None
                   skips that check, same as U's own defaults.
    """
    X, Y = _build_grid(n_pixels)

    cmap = LinearSegmentedColormap.from_list("ellipses", CUSTOM_PALETTE, N=max(len(paths), 1))

    fig, ax = plt.subplots(figsize=(7, 9))
    _draw_static_panel(ax, panel_number)

    for k, path_number in enumerate(paths):
        color = cmap(k / max(len(paths) - 1, 1))
        path_idx = path_number - 1

        U_arr = np.zeros_like(X)
        U(U_arr, (X, Y), panel_number=panel_number, state=state,
          path_indices=[path_idx], beta_constant=beta)

        # The W > 0 region's boundary is exactly the path's ellipse of influence.
        ax.contour(X, Y, U_arr, levels=[1e-9], colors=[color], linewidths=2.0)

        a, b = SENSOR_PAIRS[path_idx]
        p1, p2 = SENSOR_POSITIONS[a - 1], SENSOR_POSITIONS[b - 1]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=1.2,
                linestyle="--", alpha=0.8, zorder=2)
        ax.plot([], [], color=color, linewidth=2.0, label=f"Path {path_number} (S{a}-S{b})")

    ax.set_aspect("equal")
    ax.invert_xaxis()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Ellipse of influence — beta={beta}")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    BETAS = [ 50, 100, 500, 1000]

    for beta in BETAS:
        plot_ellipses_of_influence([1, 7, 15, 22], beta=beta, panel_number=109, n_pixels=5000,
                                    save_path=f"ellipse_of_influence_beta{beta}.svg")
