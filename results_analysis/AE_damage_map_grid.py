# This code generates heatmap on the panel using sHI from the autoencoder

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

from imagining_alghoritm import P_AE, U, WCPDI
from path_performance import load_cached, average_sHI_over_frequency, OUT_DIR, PANELS
from plot_panel import _draw_static_panel
from config import DEFAULT_N_PIXELS, CMAP_HEATMAP, PANEL_H, PANEL_W, TEST_PANEL, PANELS


def _load_sHI_avg(panel_number, fold, sHI=None):
    '''sHI averaged over the 6 frequencies for every path, from path_performance.py's cache. Outputs a list of 28 arrays (one per path) of length n_states'''
    dataset = str(panel_number)
    if sHI is None:
        cached = load_cached()
        if cached is None:
            raise FileNotFoundError(
                f"No cached results in {OUT_DIR} -- run path_performance.py first."
            )
        sHI, _ = cached

    sHI_avg = average_sHI_over_frequency(sHI, dataset, fold=fold)
    n_states_seen = {len(c) for c in sHI_avg if c is not None}
    if not n_states_seen:
        raise ValueError(f"No sHI data at all for panel {dataset}, fold {fold!r}.")
    n_states = n_states_seen.pop()

    missing = [i for i, c in enumerate(sHI_avg) if c is None]
    if missing:
        print(f"Warning: no sHI data for paths {missing} (panel {dataset}, fold {fold!r}) -- treating as 0 contribution.")
    sHI_avg = [c if c is not None else np.zeros(n_states) for c in sHI_avg]
    return sHI_avg, n_states


def plot_damage_map_grid(panel_numbers=None, fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
                          fold="ensemble", n_pixels=DEFAULT_N_PIXELS, sHI=None, save_path=None):
    """5x5 grid of raw (non-normalised) WCPDI damage maps: one row per panel, one
    column per lifetime fraction (state = round(fraction * (n_states - 1)) for that
    panel). Each subplot autoscales to its own data range -- values aren't
    comparable panel-to-panel or fraction-to-fraction, only the spatial pattern is."""
    if panel_numbers is None:
        panel_numbers = PANELS

    if sHI is None:
        cached = load_cached()
        if cached is None:
            raise FileNotFoundError(f"No cached results in {OUT_DIR} -- run path_performance.py first.")
        sHI, _ = cached

    dA = (PANEL_W * PANEL_H) / n_pixels
    dx = np.sqrt(dA)
    x = np.arange(0, PANEL_W + dx, dx)
    y = np.arange(0, PANEL_H + dx, dx)
    X, Y = np.meshgrid(x, y, indexing='ij')
    grid = (X, Y)

    n_rows, n_cols = len(panel_numbers), len(fractions)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for row, panel_number in enumerate(panel_numbers):
        sHI_avg, n_states = _load_sHI_avg(panel_number, fold, sHI=sHI)
        for col, fraction in enumerate(fractions):
            ax = axes[row][col]
            state = int(round(fraction * (n_states - 1)))

            U_arr = np.zeros_like(X)
            U(U_arr, grid, panel_number=int(panel_number), state=state)

            P_arr = np.zeros_like(X)
            sHI_per_state = [curve[state] for curve in sHI_avg]
            P_AE(P_arr, grid, sHI_per_state, panel_number=int(panel_number), state=state)
            wcpdi_map = WCPDI(P_arr, U_arr)

            _draw_static_panel(ax, int(panel_number))
            im = ax.imshow(wcpdi_map.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap=CMAP_HEATMAP)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='WCPDI Value')

            if row == 0:
                ax.set_title(f"{fraction:.0%} lifetime")
            if col == 0:
                ax.set_ylabel(f"Panel {panel_number}", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])

    #fig.suptitle("WCPDI damage maps (AE) across lifetime")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format='svg', bbox_inches='tight')
    else:
        plt.show()
    return fig


if __name__ == "__main__":
    
    panel_number = int(TEST_PANEL[0])
    fold = "ensemble"

    cached = load_cached()
    if cached is None:
        raise FileNotFoundError(f"No cached results in {OUT_DIR} -- run path_performance.py first.")
    sHI, _ = cached

    x = np.linspace(0, PANEL_W, 100)
    y = np.linspace(0, PANEL_H, 100)
    grid = np.meshgrid(x, y, indexing='ij')

    sHI_avg, _ = _load_sHI_avg(panel_number, fold, sHI=sHI)
    sHI_per_state = [curve[0] for curve in sHI_avg]  # state 0, one sHI value per path

    P_values = np.zeros((len(x), len(y)))
    P_AE(P_values, grid, sHI_per_state, panel_number=panel_number, state=0)

    plt.figure(figsize=(6, 8))
    plt.imshow(P_values.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap=CMAP_HEATMAP)
    plt.colorbar(label='Damage Probability')
    plt.title('Damage Probability Map (AE)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.tight_layout()

    U_values = np.zeros((len(x), len(y)))
    U(U_values, grid, panel_number=panel_number, state=0)

    plt.figure(figsize=(6, 8))
    plt.imshow(U_values.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap=CMAP_HEATMAP)
    plt.colorbar(label='Weight Sum')
    plt.title('Weight Map')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.tight_layout()

    WCPDI_map = WCPDI(P_values, U_values)

    plt.figure(figsize=(6, 8))
    plt.imshow(WCPDI_map.T, extent=(0, PANEL_W, 0, PANEL_H), origin='lower', cmap=CMAP_HEATMAP)
    plt.colorbar(label='WCPDI Value')
    plt.title('WCPDI Map (AE)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.tight_layout()
    #plt.show()
    plot_damage_map_grid(panel_numbers=[int(p) for p in PANELS], fractions=(0.0, 0.25, 0.5, 0.75, 1.0),  n_pixels=DEFAULT_N_PIXELS, sHI=sHI, save_path=f"WCPDI_AE_damage_maps_grid.svg")