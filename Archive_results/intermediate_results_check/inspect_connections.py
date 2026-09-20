'''
This file aims at vizualizing connection between paths on the panel and the graph connectivity. The following functions are defined:
- visualize: shows the graph on the panel and the geometric connection matrix as a heatmap
- visualize_node_subgraph_abstract: shows the subgraph induced by a list of nodes, with edges colored by their geometric connection strength
- visualize_adjacency: (as visualize but with adjacency matrix) shows the weighted adjacency graph for a single state/frequency, with edges colored by their weight, and the adjacency matrix as a heatmap
- visualize_path: given a path index, shows the path on the panel and its connections to other paths
- visualize_paths_pairs: shows path pairs on the panel 
- plot_panel_schematic: draws panel geometry details (sensors positions, damage points and dimensions)
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
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx

from weight_matrix import find_crossings, adjencency_matrix, find_crossings_by_area
from config import (
    mat_file_path, DEFAULT_FREQ_INDEX, CMAP_SEQUENTIAL, CMAP_BLUES,
    SENSOR_POSITIONS, SENSOR_PAIRS, PANEL_W, PANEL_H, DAMAGE_POINTS, CUSTOM_PALETTE
)



def build_crossing_graph(by_area):
    # Build a graph based on the adjacency matrix imported from weight_matrix.py
    is_crossing = find_crossings_by_area() if by_area else find_crossings()
    G = nx.Graph()
    G.add_nodes_from(range(28))
    rows, cols = np.nonzero(np.triu(is_crossing))
    for i, j in zip(rows, cols):
        G.add_edge(int(i), int(j))
    return G, is_crossing


def node_positions(): 
    # Finds midpoints of every path, where nodes represenations are placed
    pos = {}
    for k, (s1, s2) in enumerate(SENSOR_PAIRS):
        p1 = SENSOR_POSITIONS[s1 - 1]
        p2 = SENSOR_POSITIONS[s2 - 1]
        pos[k] = (p1 + p2) / 2
    return pos

def find_path_index(sensor_1,sensor_2):
    for idx, (s1, s2) in enumerate(SENSOR_PAIRS):
        if (s1 == sensor_1 and s2 == sensor_2) or (s1 == sensor_2 and s2 == sensor_1):
            return idx
    return None

def visualize(show_path_lines=True, by_area=False):
    # show the graph on the panel and the IS_CROSSING matrix as a heatmap

    G, is_crossing = build_crossing_graph(by_area=by_area)
    pos = node_positions()                       # {node_idx: (x, y)}


    fig, axes = plt.subplots(1, 2, figsize=(16, 7)) 

    # --- left: graph drawn on the panel ---
    ax = axes[0]
    # draw panel boundary
    rect = mpatches.Rectangle((0, 0), PANEL_W, PANEL_H,
                               linewidth=1.5, edgecolor="black",
                               facecolor="whitesmoke", zorder=0)
    ax.add_patch(rect)

    # draw sensor positions
    ax.scatter(SENSOR_POSITIONS[:, 0], SENSOR_POSITIONS[:, 1],
               s=80, color="steelblue", zorder=4)
    for i, (x, y) in enumerate(SENSOR_POSITIONS, start=1):
        ax.text(x, y + 0.006, f"S{i}", ha="center", va="bottom",
                fontsize=8, color="steelblue", zorder=5)

    # faint path lines for context
    if show_path_lines:
        for s1, s2 in SENSOR_PAIRS:
            p1, p2 = SENSOR_POSITIONS[s1 - 1], SENSOR_POSITIONS[s2 - 1]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color="lightgray", linewidth=0.8, zorder=1)

    # draw edges (crossing connections)
    for i, j in G.edges():
        xi, yi = pos[i]
        xj, yj = pos[j]
        ax.plot([xi, xj], [yi, yj], color="tomato", linewidth=1.0,
                alpha=0.6, zorder=2)

    # draw nodes (path midpoints)
    node_x = [pos[k][0] for k in range(28)]
    node_y = [pos[k][1] for k in range(28)]
    ax.scatter(node_x, node_y, s=60, color="darkorange",
               edgecolors="black", linewidths=0.5, zorder=3)
    for k in range(28):
        ax.text(pos[k][0], pos[k][1] + 0.005, str(k + 1),
                ha="center", va="bottom", fontsize=6, color="black", zorder=6)

    margin = 0.01
    ax.set_xlim(-margin, PANEL_W + margin)
    ax.set_ylim(-margin, PANEL_H + margin)
    ax.set_aspect("equal")
    ax.invert_xaxis()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Crossing graph on panel\n"
        f"{G.number_of_nodes()} nodes · {G.number_of_edges()} edges"
    )

    # legend
    ax.scatter([], [], s=60, color="darkorange", edgecolors="black",
               linewidths=0.5, label="Path node (midpoint)")
    ax.plot([], [], color="tomato", linewidth=1.0, label="Crossing edge")
    ax.plot([], [], color="lightgray", linewidth=0.8, label="Sensor path")
    ax.legend(loc="upper right", fontsize=8)

    # --- right: IS_CROSSING matrix as a heatmap ---
    ax2 = axes[1]
    im = ax2.imshow(is_crossing.astype(float), cmap=CMAP_BLUES, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax2, ticks=[0, 1], label="Crosses (1 = yes)", shrink=0.8)
    tick_labels = [str(i + 1) for i in range(28)]
    ax2.set_xticks(range(28))
    ax2.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax2.set_yticks(range(28))
    ax2.set_yticklabels(tick_labels, fontsize=7)
    ax2.set_xlabel("Path index")
    ax2.set_ylabel("Path index")
    ax2.set_title("IS_CROSSING matrix")

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Avg degree: {2 * G.number_of_edges() / G.number_of_nodes():.1f}")

    fig.tight_layout()
    plt.show()


def visualize_node_subgraph_abstract(nodes, by_area=False):
    """
    Abstract-only layout (no panel view) of connections between the nodes from the argument,
    edges colored by their actual is_crossing strength 

    nodes : list of path numbers to plot, 1-indexed (1-28).
    by_area : bool, if True, nodes are positioned by their area.
    """
    G, is_crossing_matrix = build_crossing_graph(by_area=by_area)
    node_idx = [n - 1 for n in nodes]
    subG = G.subgraph(node_idx)

    layout = nx.circular_layout(subG)
    labels = {n: str(n + 1) for n in subG.nodes()}

    weights = [is_crossing_matrix[i, j] for i, j in subG.edges()]
    for i, j in subG.edges():
        print(f"Connection between nodes {i + 1} and {j + 1}: {is_crossing_matrix[i, j]}")
    
    base_cmap = plt.get_cmap(CMAP_BLUES)
    cmap = LinearSegmentedColormap.from_list(
        "blues_visible", base_cmap(np.linspace(0.35, 1.0, 256))
    )
    wmin = min(weights) if weights else 0.0
    wmax = max(weights) if weights else 1.0
    norm = plt.Normalize(vmin=wmin, vmax=wmax if wmax > wmin else wmin + 1e-9)
    edge_colors = [cmap(norm(w)) for w in weights]

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(subG, layout, ax=ax, edge_color=edge_colors, width=2.2)
    nx.draw_networkx_nodes(subG, layout, ax=ax, node_color="darkorange",
                           node_size=550, edgecolors="black", linewidths=0.8)
    nx.draw_networkx_labels(subG, layout, labels=labels, ax=ax,
                            font_size=9, font_color="black", font_weight="bold")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Crossing strength (is_crossing)", shrink=0.8)

    ax.set_aspect("equal")
    ax.axis("off")
    #ax.set_title(f"Subgraph induced by nodes {nodes} ")
    fig.tight_layout()
    plt.show()
    return subG


def build_adjacency_graph(States, state_idx, freq_idx, by_area=False):
    """Weighted crossing graph for one state/frequency: edges only between
    crossing paths, weighted by adjencency_matrix (attention + crossing)."""
    type = "by_area" if by_area else "basic"
    adjacency = adjencency_matrix(States, state_idx, freq_idx, type=type)
    if adjacency.ndim != 2:
        raise ValueError(
            f"visualize_adjacency needs a single state (got state_idx={state_idx!r}, "
            f"which produced a {adjacency.ndim}D array). Pass one GLOBAL state index."
        )
    G = nx.Graph()
    G.add_nodes_from(range(28))
    rows, cols = np.nonzero(np.triu(adjacency))
    for i, j in zip(rows, cols):
        G.add_edge(int(i), int(j), weight=float(adjacency[i, j]))
    return G, adjacency


def visualize_adjacency(States, state_idx, freq_idx, show_path_lines=True, plain=False, by_area=False):
    """
    Visualizes the weighted adjacency graph from weight_matrix.adjencency_matrix:
      left  = graph drawn on the panel, edges colored by weight
      right = adjacency matrix heatmap

      if plain=True, only the adjacency matrix is shown (no panel view).

    state_idx : a single state index (int), not ":" — the adjacency
                matrix is only 2D (drawable as one graph) for a single state.
    """
    G, adjacency = build_adjacency_graph(States, state_idx, freq_idx, by_area=by_area)
    pos = node_positions()

    if not plain:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # --- left: weighted graph drawn on the panel ---
        ax = axes[0]
        rect = mpatches.Rectangle((0, 0), PANEL_W, PANEL_H,
                                linewidth=1.5, edgecolor="black",
                                facecolor="whitesmoke", zorder=0)
        ax.add_patch(rect)

        # sensor positions
        ax.scatter(SENSOR_POSITIONS[:, 0], SENSOR_POSITIONS[:, 1],
                s=80, color="steelblue", zorder=4)
        for i, (x, y) in enumerate(SENSOR_POSITIONS, start=1):
            ax.text(x, y + 0.006, f"S{i}", ha="center", va="bottom",
                    fontsize=8, color="steelblue", zorder=5)

        # faint path lines for context
        if show_path_lines:
            for s1, s2 in SENSOR_PAIRS:
                p1, p2 = SENSOR_POSITIONS[s1 - 1], SENSOR_POSITIONS[s2 - 1]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color="lightgray", linewidth=0.8, zorder=1)

        # weighted edges (color + thickness scale with adjacency weight)
        weights = [G[i][j]["weight"] for i, j in G.edges()]
        cmap = plt.get_cmap("plasma")
        wmin = min(weights) if weights else 0.0
        wmax = max(weights) if weights else 1.0
        norm = plt.Normalize(vmin=wmin, vmax=wmax if wmax > wmin else wmin + 1e-9)

        for i, j in G.edges():
            w = G[i][j]["weight"]
            xi, yi = pos[i]
            xj, yj = pos[j]
            ax.plot([xi, xj], [yi, yj], color=cmap(norm(w)),
                    linewidth=0.8 + 2.5 * norm(w), alpha=0.8, zorder=2)

        # nodes (path midpoints)
        node_x = [pos[k][0] for k in range(28)]
        node_y = [pos[k][1] for k in range(28)]
        ax.scatter(node_x, node_y, s=60, color="darkorange",
                edgecolors="black", linewidths=0.5, zorder=3)
        for k in range(28):
            ax.text(pos[k][0], pos[k][1] + 0.005, str(k + 1),
                    ha="center", va="bottom", fontsize=6, color="black", zorder=6)

        margin = 0.01
        ax.set_xlim(-margin, PANEL_W + margin)
        ax.set_ylim(-margin, PANEL_H + margin)
        ax.set_aspect("equal")
        ax.invert_xaxis()
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(
            f"Adjacency graph on panel (state {state_idx}, freq {freq_idx})\n"
            f"{G.number_of_nodes()} nodes · {G.number_of_edges()} edges"
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Edge weight (attention + crossing)", shrink=0.8)

        # legend
        ax.scatter([], [], s=60, color="darkorange", edgecolors="black",
                linewidths=0.5, label="Path node (midpoint)")
        ax.plot([], [], color="lightgray", linewidth=0.8, label="Sensor path")
        ax.legend(loc="upper right", fontsize=8)

        # --- right: adjacency matrix as a heatmap ---
        ax2 = axes[1]
        im = ax2.imshow(adjacency, cmap=CMAP_SEQUENTIAL)
        fig.colorbar(im, ax=ax2, label="Adjacency weight", shrink=0.8)
        tick_labels = [str(i + 1) for i in range(28)]
        ax2.set_xticks(range(28))
        ax2.set_xticklabels(tick_labels, rotation=90, fontsize=7)
        ax2.set_yticks(range(28))
        ax2.set_yticklabels(tick_labels, fontsize=7)
        ax2.set_xlabel("Path index")
        ax2.set_ylabel("Path index")
        ax2.set_title("Adjacency matrix")

        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Edges: {G.number_of_edges()}")
        if G.number_of_edges():
            print(f"Weight range: {wmin:.4f} to {wmax:.4f}")
    else:
        wmin = np.min(adjacency)
        wmax = np.max(adjacency)
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(adjacency, cmap=CMAP_SEQUENTIAL, vmin=wmin, vmax=wmax)
        

    fig.tight_layout()
    plt.show()


def _draw_path_panel(ax, path_number, neighbours, p1_t, p2_t):
    """Draw the panel view for a single path onto ax."""
    ax.add_patch(mpatches.Rectangle((0, 0), PANEL_W, PANEL_H,
                                    linewidth=1.5, edgecolor="black",
                                    facecolor="whitesmoke", zorder=0))
    for sa, sb in SENSOR_PAIRS:
        pa, pb = SENSOR_POSITIONS[sa - 1], SENSOR_POSITIONS[sb - 1]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                color="lightgray", linewidth=0.8, zorder=1)

    for n in neighbours:
        sn1, sn2 = SENSOR_PAIRS[n]
        pn1, pn2 = SENSOR_POSITIONS[sn1 - 1], SENSOR_POSITIONS[sn2 - 1]
        ax.plot([pn1[0], pn2[0]], [pn1[1], pn2[1]],
                color="darkorange", linewidth=2.0, zorder=2)
        mx, my = (pn1 + pn2) / 2
        ax.text(mx, my + 0.005, str(n + 1),
                ha="center", va="bottom", fontsize=7, color="darkorange", zorder=5)

    ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]],
            color="steelblue", linewidth=2.5, zorder=3)
    mx_t, my_t = (p1_t + p2_t) / 2
    ax.text(mx_t, my_t + 0.005, str(path_number),
            ha="center", va="bottom", fontsize=8,
            fontweight="bold", color="steelblue", zorder=6)

    ax.scatter(SENSOR_POSITIONS[:, 0], SENSOR_POSITIONS[:, 1],
               s=60, color="black", zorder=4)
    for i, (x, y) in enumerate(SENSOR_POSITIONS, start=1):
        ax.text(x, y + 0.006, f"S{i}", ha="center", va="bottom",
                fontsize=7, color="black", zorder=7)

    margin = 0.001
    ax.set_xlim(-margin, PANEL_W + margin)
    ax.set_ylim(-margin, PANEL_H + margin)
    ax.set_aspect("equal")
    ax.invert_xaxis()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    #ax.set_title("Panel view")
    ax.plot([], [], color="steelblue", linewidth=2.5, label=f"Path {path_number}")
    ax.plot([], [], color="darkorange", linewidth=2.0,
            label=f"Crossing paths ({len(neighbours)})")
    ax.plot([], [], color="lightgray", linewidth=0.8, label="Other paths")
    ax.legend(loc="upper right", fontsize=9)


def _draw_ego_graph(ax, G, idx, neighbours, path_number):
    """Draw the star ego-graph onto ax."""
    from matplotlib.lines import Line2D

    ego_nodes = [idx] + neighbours
    ego_G     = G.subgraph(ego_nodes)
    n_nb      = len(neighbours)
    angles    = np.linspace(0, 2 * np.pi, n_nb, endpoint=False)
    star_pos  = {idx: np.array([0.0, 0.0])}
    for nb, angle in zip(neighbours, angles):
        star_pos[nb] = np.array([np.cos(angle), np.sin(angle)])

    node_colors = ["steelblue" if n == idx else "darkorange" for n in ego_nodes]
    node_labels = {n: str(n + 1) for n in ego_nodes}

    nx.draw_networkx_edges(ego_G, star_pos, ax=ax,
                           edge_color="gray", width=1.5, alpha=0.7)
    nx.draw_networkx_nodes(ego_G, star_pos, ax=ax,
                           nodelist=ego_nodes, node_color=node_colors,
                           node_size=600, edgecolors="black", linewidths=0.8)
    nx.draw_networkx_labels(ego_G, star_pos, labels=node_labels, ax=ax,
                            font_size=9, font_color="white", font_weight="bold")

    ax.set_aspect("equal")
    ax.axis("off")
    #ax.set_title(f"Ego-graph for path {path_number}  ({n_nb} neighbours)")
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=10, label=f"Path {path_number}"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="darkorange",
               markersize=10, label="Crossing neighbour"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12)


def visualize_path(path_number):
    """
    Renders three figures for a single path:
      1. Combined — panel view (left) + ego-graph (right) side by side.
      2. Panel view alone.
      3. Ego-graph alone.

    path_number : 1-indexed (1–28).
    """
    G, _ = build_crossing_graph(by_area=False)
    idx        = path_number - 1
    neighbours = sorted(G.neighbors(idx))
    s1_t, s2_t = SENSOR_PAIRS[idx]
    p1_t = SENSOR_POSITIONS[s1_t - 1]
    p2_t = SENSOR_POSITIONS[s2_t - 1]
    suptitle = f"Path {path_number}  (S{s1_t}–S{s2_t})  ·  {len(neighbours)} crossing paths"

    # ── 1. combined figure ────────────────────────────────────────────────
    fig, (ax_panel, ax_graph) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(suptitle, fontsize=12)
    _draw_path_panel(ax_panel, path_number, neighbours, p1_t, p2_t)
    _draw_ego_graph(ax_graph, G, idx, neighbours, path_number)
    fig.tight_layout()

    # ── 2. panel view alone ───────────────────────────────────────────────
    fig_panel, ax_p = plt.subplots(figsize=(7, 9))
    #fig_panel.suptitle(suptitle, fontsize=11)
    _draw_path_panel(ax_p, path_number, neighbours, p1_t, p2_t)
    fig_panel.tight_layout()

    # ── 3. ego-graph alone ────────────────────────────────────────────────
    fig_ego, ax_g = plt.subplots(figsize=(7, 7))
    #fig_ego.suptitle(suptitle, fontsize=11)
    _draw_ego_graph(ax_g, G, idx, neighbours, path_number)
    fig_ego.tight_layout()

    plt.show()



def _draw_panel_background(ax):
    """Draw the empty panel (border + faint path lines + sensor markers) onto ax."""
    ax.add_patch(mpatches.Rectangle((0, 0), PANEL_W, PANEL_H,
                                    linewidth=1.5, edgecolor="black",
                                    facecolor="whitesmoke", zorder=0))
    for sa, sb in SENSOR_PAIRS:
        pa, pb = SENSOR_POSITIONS[sa - 1], SENSOR_POSITIONS[sb - 1]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                color="lightgray", linewidth=0.8, zorder=1)
    ax.scatter(SENSOR_POSITIONS[:, 0], SENSOR_POSITIONS[:, 1],
               s=60, color="black", zorder=4)
    for i, (x, y) in enumerate(SENSOR_POSITIONS, start=1):
        ax.text(x, y + 0.006, f"S{i}", ha="center", va="bottom",
                fontsize=7, color="black", zorder=7)


def _highlight_path_pair(ax, path_a, path_b, color):
    """Draw both paths of one pair in a shared color onto ax (no background)."""
    label_offset = 0.006
    for p in (path_a, path_b):
        s1, s2 = SENSOR_PAIRS[p - 1]
        p1, p2 = SENSOR_POSITIONS[s1 - 1], SENSOR_POSITIONS[s2 - 1]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2.5, zorder=3)
        mid = (p1 + p2) / 2

        direction = p2 - p1
        norm = np.linalg.norm(direction)
        perp = np.array([-direction[1], direction[0]]) / norm if norm > 0 else np.array([0.0, 1.0])
        if perp[1] < 0:  # keep labels biased toward the top, for consistency
            perp = -perp
        label_pos = mid + perp * label_offset

        ax.text(label_pos[0], label_pos[1], str(p), ha="center", va="center",
                fontsize=12, fontweight="bold", color=color, zorder=6)


def visualize_paths_pairs(path_pairs):
    """
    Highlights several path pairs on one shared panel plot -- each pair drawn in its
    own color (both paths of a pair share that color), so pairs are visually distinct.

    path_pairs : list of (path_a, path_b), 1-indexed (1-28).
    """
    _, is_crossing_matrix = build_crossing_graph(by_area=False)

    fig, ax = plt.subplots(figsize=(7, 9))
    _draw_panel_background(ax)

    cmap = LinearSegmentedColormap.from_list("panel_pairs", CUSTOM_PALETTE, N=max(len(path_pairs), 1))
    eps = 10e-6

    for k, (path_a, path_b) in enumerate(path_pairs):
        color = cmap(k / max(len(path_pairs) - 1, 1))
        idx_a, idx_b = path_a - 1, path_b - 1
        crossing = bool(is_crossing_matrix[idx_a, idx_b] > eps)

        _highlight_path_pair(ax, path_a, path_b, color)
        ax.plot([], [], color=color, linewidth=2.5,
                label=f"{path_a} & {path_b} ")

    margin = 0.01
    ax.set_xlim(-margin, PANEL_W + margin)
    ax.set_ylim(-margin, PANEL_H + margin)
    ax.set_aspect("equal")
    ax.invert_xaxis()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    #ax.set_title(f"{len(path_pairs)} path pairs highlighted on panel")
    ax.plot([], [], color="lightgray", linewidth=0.8, label="Other paths")
    ax.legend(loc="upper right", fontsize=12)

    fig.tight_layout()
    plt.show()
    


def plot_panel_schematic(impact_points=None, ax=None, title="Panel schematic",
                         show_dims=False, save_path=None):
    """
    Detailed panel schematic with sensors, stiffeners, disbond area and optional
    impact points — styled to sit alongside a MATLAB damage-map frame.

    impact_points : dict  {'label': np.array([x, y])}  positions in metres, or None.
    ax            : existing Axes to draw on, or None to create a new figure.
    show_dims     : draw span arrows with mm labels outside the panel (default True).
    save_path     : file path to save the figure (e.g. 'panel.svg').  None = no save.
    """
    # ── geometry constants ────────────────────────────────────────────────
    STIFFENERS = [(PANEL_W/2, 0.03)]   # (x_centre, half_width) m -- spans -50mm to +50mm from center
    X_GRID = [0.0, 0.025, 0.065, 0.100, 0.140, PANEL_W]
    Y_GRID = [0.0, 0.080, 0.160, PANEL_H]
    X_DIMS = [25, 40, 35, 40, 25]
    Y_DIMS = [80, 80, 80]
    SENSOR_R = 0.0085
    IMPACT_R = 0.0085

    # ── figure setup ──────────────────────────────────────────────────────
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(4, 5.5))
    else:
        fig = ax.get_figure()

    # Panel background
    ax.add_patch(mpatches.Rectangle(
        (0, 0), PANEL_W, PANEL_H,
        linewidth=1.5, edgecolor="black", facecolor="white", zorder=0,
    ))

    # Stiffeners (light-blue vertical bands)
    for x_c, hw in STIFFENERS:
        ax.add_patch(mpatches.Rectangle(
            (x_c - hw, 0), 2 * hw, PANEL_H,
            linewidth=0, facecolor=CUSTOM_PALETTE[2], alpha=0.55, zorder=1,
        ))
        # Boundary markers, labeled as mm offset from the panel center (distinct
        # from the x-tick labels, which give absolute position from the left edge).
        for edge_x, offset_mm in ((x_c - hw, -hw * 1000), (x_c + hw, hw * 1000)):
            ax.axvline(edge_x, color=CUSTOM_PALETTE[0], linestyle=":", linewidth=0.8, zorder=2)
            
        

    
    # Internal dashed grid lines
    for xg in X_GRID[1:-1]:
        ax.axvline(xg, color="dimgray", linestyle="--", linewidth=0.6, zorder=2)
    for yg in Y_GRID[1:-1]:
        ax.axhline(yg, color="dimgray", linestyle="--", linewidth=0.6, zorder=2)

   

    # Disbond area (panel 123)
    dp = DAMAGE_POINTS.get(123)
    if dp is not None and dp.ndim == 2:
        x_min, y_min = dp.min(axis=0)
        x_max, y_max = dp.max(axis=0)
        ax.add_patch(mpatches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=1.5, edgecolor="black", facecolor=CUSTOM_PALETTE[1],
            alpha=0.85, zorder=3,
        ))
        ax.text((x_min + x_max) / 2, (y_min + y_max) / 2, "23",
                ha="center", va="center", fontsize=10,
                color="white", fontweight="bold", zorder=4)

    # Sensors
    for i, (x, y) in enumerate(SENSOR_POSITIONS, start=1):
        ax.add_patch(plt.Circle((x, y), SENSOR_R, color=CUSTOM_PALETTE[0], zorder=5))
        ax.text(x, y, str(i), ha="center", va="center",
                fontsize=10, color="white", fontweight="bold", zorder=6)

    # Impact points
    if impact_points:
        for label, pos_m in impact_points.items():
            label = "0"+str(label%100)
            ax.add_patch(plt.Circle((pos_m[0], pos_m[1]), IMPACT_R,
                                    color=CUSTOM_PALETTE[1], zorder=5))
            ax.text(pos_m[0], pos_m[1], label, ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold", zorder=6)

    # ── axes ──────────────────────────────────────────────────────────────
    INNER  = 0.003                              # gap inside axes limits
    OUTER  = 0.022 if show_dims else INNER      # space for arrows + labels

    ax.set_xlim(-INNER, PANEL_W + OUTER)
    ax.set_ylim(-OUTER, PANEL_H + INNER)
    ax.set_aspect("equal")
    ax.invert_xaxis()

    ax.set_xticks(X_GRID)
    ax.set_xticklabels([str(int(round(x * 1000))) for x in X_GRID], fontsize=15)
    ax.set_xlabel("x (mm)", fontsize=15, labelpad=2)
    ax.set_yticks(Y_GRID)
    ax.set_yticklabels([str(int(round(y * 1000))) for y in Y_GRID], fontsize=15)
    ax.set_ylabel("y (mm)", fontsize=15, labelpad=2)
    ax.tick_params(length=2, pad=2)

    # Dimension span annotations (optional)
    if show_dims:
        ann_kw = dict(arrowstyle="<->", color="black", lw=0.8)
        y_ann = -OUTER * 0.55

        for x0, x1, lbl in zip(X_GRID, X_GRID[1:], X_DIMS):
            ax.annotate("", xy=(x0, y_ann), xytext=(x1, y_ann),
                        arrowprops=ann_kw, annotation_clip=False)
            ax.text((x0 + x1) / 2, y_ann - 0.004, str(lbl),
                    ha="center", va="top", fontsize=15, clip_on=False)

        x_ann = PANEL_W + OUTER * 0.55
        for y0, y1, lbl in zip(Y_GRID, Y_GRID[1:], Y_DIMS):
            ax.annotate("", xy=(PANEL_W, y0), xytext=(PANEL_W, y1),
                        arrowprops=ann_kw, annotation_clip=False)
            ax.text(x_ann + 0.001, (y0 + y1) / 2, str(lbl),
                    ha="left", va="center", fontsize=15, clip_on=False)


    if standalone:
        fig.tight_layout(pad=0.4)
        if save_path is not None:
            fig.savefig(save_path, format="svg", bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()


if __name__ == "__main__":

    
    impact_pts = {name: xy for name, xy in DAMAGE_POINTS.items() if name != 123}
    plot_panel_schematic(impact_points=impact_pts, title="Panel 123",
                             save_path="panel_scheme.svg")

    
    fig, ax = plt.subplots(figsize=(7, 9))
    _draw_panel_background(ax)

    # find 5 the largest values in the crossing and their indecies
    c = find_crossings_by_area()
    c_upper = np.triu(c, k=1)  # zero out lower triangle + diagonal, keep each pair once
    flat_idx = np.argsort(c_upper, axis=None)[-5:][::-1]  # 5 distinct pairs, strongest first
    top = np.unravel_index(flat_idx, c_upper.shape)

    pairs = [(top[0][0]+1, top[1][0]+1), (4, find_path_index(4,5)+1),(find_path_index(8,5)+1,find_path_index(6,7)+1) ]
    pairs = [(25,26),(19,4),(9,28)]
    visualize_paths_pairs(pairs)
    visualize_node_subgraph_abstract([25,26,19,4,9,28])

    visualize_node_subgraph_abstract([top[0][0]+1, top[1][0]+1, 4, find_path_index(4,5)+1,find_path_index(8,5)+1,find_path_index(6,7)+1])

    visualize_path(6)

    visualize() 
    
    from states import states
    st_123_43 = states(str(mat_file_path("123_43")))
    visualize_adjacency(st_123_43, state_idx=70, freq_idx=DEFAULT_FREQ_INDEX, plain=True)

    
    
    