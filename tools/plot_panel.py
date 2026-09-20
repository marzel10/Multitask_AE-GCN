'''
This file defines function to visualize the PZT panel,plot_panel_with_paths: draw the panel with sensors, damage point, and optionally active paths

The panel geometry and sensor positions are defined as constants at the top of the file, matching the experimental setup.

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


import matplotlib.patches as mpatches


from config import PANEL_W, PANEL_H, SENSOR_POSITIONS, DAMAGE_POINTS 


def _draw_static_panel(ax, panel_number):
    """Draw panel boundary, sensors, and damage point onto ax."""
    rect = mpatches.Rectangle(
        (0, 0), PANEL_W, PANEL_H,
        linewidth=1.5, edgecolor="black", facecolor="whitesmoke", zorder=0,
    )
    ax.add_patch(rect)

    ax.scatter(
        SENSOR_POSITIONS[:, 0], SENSOR_POSITIONS[:, 1],
        s=60, color="steelblue", zorder=3,
    )
    for i, (x, y) in enumerate(SENSOR_POSITIONS, start=1):
        ax.text(x, y + 0.006, str(i), ha="center", va="bottom",
                fontsize=8, color="steelblue", zorder=4)

    dp = DAMAGE_POINTS.get(panel_number)
    if dp is not None and dp.ndim == 1:
        ax.scatter(dp[0], dp[1], s=120, color="green", marker="X", zorder=5)
        ax.text(dp[0], dp[1] + 0.008, "Impact",
                ha="center", va="bottom", fontsize=8, color="green", zorder=6)
    elif dp is not None and dp.ndim == 2:
        x_min, y_min = dp.min(axis=0)
        x_max, y_max = dp.max(axis=0)
        width, height = x_max - x_min, y_max - y_min
        rect = mpatches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=1.5, edgecolor="green", facecolor="none", linestyle="--", zorder=5,
            label="Debond area",
        )
        ax.add_patch(rect)
    margin = 0.001
    ax.set_xlim(-margin, PANEL_W + margin)
    ax.set_ylim(-margin, PANEL_H + margin)
    ax.invert_xaxis()
    ax.set_aspect("equal")
    

