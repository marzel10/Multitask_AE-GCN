"""
Central configuration for the SHM pipeline.
"""

import os
from itertools import combinations
from pathlib import Path

import numpy as np

# ===========================================================================
# Paths
# ===========================================================================
PROJECT_ROOT = Path(__file__).resolve().parent  # .../GAN/

TEST_PANEL = [os.environ.get("SHM_TEST_PANEL", "103")]  # <-- change the default here, or set the SHM_TEST_PANEL env var, to make a different panel the held-out test panel
TEST_RUN_DIR = PROJECT_ROOT / f"test_{TEST_PANEL[0]}_wo123"  # every result that depends on which panel is TEST_PANEL lives under here, so different TEST_PANEL runs never collide
DEBUG_MODE = False

# Data directories -- shared/reusable across every TEST_PANEL run, not nested under TEST_RUN_DIR
DATA_DIR = PROJECT_ROOT / "data"                                                # raw states directory used by mat_file_path()
FEATURES_CACHE_DIR = PROJECT_ROOT / "features_cache"                            # create_datastores.py saves there pre computed features from raw states (AE input) -- a function of panel/freq/path only, same regardless of TEST_PANEL

# Everything below depends on TEST_PANEL (training uses BASE_PANELS, which
# depends on it), so it's nested under TEST_RUN_DIR.
GRAPH_DATA_DIR = TEST_RUN_DIR / "graph_data"                                    # PyG dataset root (raw/ + processed/). It is therefore used by every program that uses GNN (GCN input)
BO_SEARCH_RESULTS_DIR = TEST_RUN_DIR / "results"                                # Directory for the BO_GCN search results, folders with different types and frequencis are inside
# Analysis results directories
AE_RESULTS_DIR = TEST_RUN_DIR / "path_performance_results"                      # Written by: path_performance.py, summary plots and HI and metrics arrays are saved there.
GCN_RESULTS_DIR = TEST_RUN_DIR / "graph_performance_results_basic"              # Written by: graph_performance.py, summary plots and HI and metrics arrays are saved there.
BETA_SWEEP_RESULTS_DIR = TEST_RUN_DIR / "graph_performance_results_beta_sweep"  # Written by: graph_performance_beta_sweep.py, summary plots and HI and metrics arrays are saved there.
COMPARE_OUT_DIR = TEST_RUN_DIR / "damage_loss_evaluation_results"               # Written by: Damage_metric_summary.py, summary plots and metrics arrays are saved there.
# Other used in main workflow but  particularly relevant
MODEL_DATABASE_XLSX_FEATURES = TEST_RUN_DIR / "model_database_features.xlsx"    # Path to the AE model database
BO_TUNER_DIR = TEST_RUN_DIR / "tuner_dir"                                       # keras-tuner scratch dir.
BO_RESULTS_DIR = TEST_RUN_DIR / "BO_results"                                    # dafault directory to save the results of the BO_AE.py (not used)
# Only created when running AE_train.py main function:
CROSS_VALIDATION_RESULTS_AE_DIR = TEST_RUN_DIR / "Cross_Validation_Results_AE"  # Cross validation AE results are saved there, used only by AE_train.py
MODEL_TRAIN_RESULTS_DIR = TEST_RUN_DIR / "model_train_results"                  # AE_train.py saves there per fold results
# Only created when running GCN_train.py main function:
CROSS_VALIDATION_RESULTS_DIR = TEST_RUN_DIR / "Cross_Validation_Results"        # Cross validation GCN results are saved there, used only by GCN_train.py
GCN_MODELS_DIR = TEST_RUN_DIR / "GCN_models"                                    # Singular GCN models will be saved there, used only by GCN_train.py

def mat_file_path(panel_name: str) -> Path:
    return DATA_DIR / f"States_{panel_name}.mat"


# ===========================================================================
# Panels
# ===========================================================================
SINGLE_FILE_PANELS = ["103", "104", "105", "109"]  # panels with one whole-panel .mat file each -- "123" has no such file, only its 8 subpanel files (see PANEL_123_SUBPANELS), so this list never changes with TEST_PANEL
ALL_BASE_PANELS = SINGLE_FILE_PANELS 
BASE_PANELS = [p for p in ALL_BASE_PANELS if p not in TEST_PANEL]  # TEST_PANEL is defined up in the Paths section, since TEST_RUN_DIR needs it first

PANEL_123_SUBPANELS = ["123_1", "123_2", "123_31", "123_32", "123_41", "123_42", "123_43", "123_44"]

TRAIN_PANELS = ["103", "104", "105"]                          # used only as deafult train panels in AE_train, never in the actual workflow
VAL_PANELS = ["103"] if TEST_PANEL[0]!="103" else ["109"]                                         # Used by: BO_AE.py for th validation inside BO , big_train.py (val_ds_names default), results_viz.py (val_ds_names)
TEST_PANELS = PANEL_123_SUBPANELS                             # Used by: BO.py (TEST_DS_NAMES), big_train.py (test_ds_names default), results_viz.py (test_ds_names) -- all 8 "123" subpanels held out as the test set
VAL_123_SUBPANELS = ["123_1", "123_31", "123_41", "123_43"]   # states_check.py val split (finer split: half of the 123 subpanels)
TEST_123_SUBPANELS = ["123_2", "123_32", "123_42", "123_44"]  # states_check.py test split (finer split: the other half of the 123 subpanels)
CV_PANELS = BASE_PANELS
CV_PANELS_INT = [int(p) for p in CV_PANELS]
FOLD_KEYS = BASE_PANELS + ["ensemble"]

# Per-123-subpanel saved-state counts / global start offsets.
# Previously byte-for-byte duplicated in weight_matrix.py and graph_dataset.py.
NR_SAVED_STATES = {"123_1": 16, "123_2": 7, "123_31": 16, "123_32": 13, "123_41": 10, "123_42": 10, "123_43": 10, "123_44": 10}
STATE_START_INDICES = {"123_1": 0, "123_2": 16, "123_31": 23, "123_32": 39, "123_41": 52, "123_42": 62, "123_43": 72, "123_44": 82}

TOTAL_STATES = {"103": 32, "104": 58, "105": 30, "109": 28, "123": 92}  # Owned by: GCN_train.py

# Sensor-failure dictionary: {panel: [failure_state, sensor_idx]}
FAILURES_RECORD = {
    "123_42": [9, 3],
    "104": [15, 8],
    "105": [16, 8],
}

VALIDATION_PANEL_MAP = {TEST_PANEL[0]: "109"}  # Owned by: extract_shi.py

# ===========================================================================
# Frequency / sampling
# ===========================================================================
DEFAULT_FREQ_INDEX = 1  # Hardcoded as freq=5 independently in: damage_map.py, extract_shi.py, heatmap_AE.py, states_check.py, states_plot.py, features_extractor.py, results_viz.py, big_train.py, GCN_train.py, graph_dataset.py, imagining_alghoritm.py, make_animations.py, prognostic_criteria.py, visualize_crossing_graph.py
FREQUENCY_MAPPING = [50,100,125,150,200,250] # [kHz]

# ==========================================================================
# Sensitivity study constants
# ==========================================================================
TYPES = [ "peak"]#, "geometry_only", "peak_tff", "peak_fft", "peak_tft", "peak_and_area", "peak_only"]  # GCN adjacency-matrix types, same as sweep_over_options.py
FREQ_FOR_BETA_SWEEP = 1  
BETAS = [25, 75, 100, 250, 500, 750, 1000, 2000, 5000]

OPTIMIZED_TYPES = [ "peak", "peak_and_area", "peak_only", "geometry_only", "peak_tff", "peak_fft", "peak_tft"]  # GCN adjacency-matrix types to run the BO optimization on. The others are skipped.



OPTIMIZE_RAW = True  # Whether to optimize the GCN with raw features (no AE) as well
LIFETIME_FRACTIONS = [0, 0.25, 0.5, 0.75, 1.0]  # Used by: Damage_metric_summary.py (compare_life_fractions)
# ===========================================================================
# Damage metric evaluation constants
# ===========================================================================
GCN_TYPES = TYPES
MODEL_TYPES = GCN_TYPES + ["path", "raw"]
TYPES_LABELS = {  "peak": "CAE-GCN", "path": "CAE", "raw": "GCN", "peak_and_area": "GCN-PA", "peak_only": "GCN-P", "geometry_only": "GCN-I", "peak_tff": r"GCN-$\mathcal{L}_{loc}$", "peak_fft": r"GCN-$\mathcal{L}_{HI}$", "peak_tft": r"GCN-$\mathcal{L}_{loc}$+$\mathcal{L}_{HI}$"}
PANELS = [int(p) for p in BASE_PANELS] + [int(TEST_PANEL[0])]

DAMAGE_MAP_N_PIXELS = 40000

# ===========================================================================
# Fitness metric evaluation constants
# ===========================================================================
GRAPH_TYPES = ["peak"] #[ "peak_tff", "peak_fft", "peak_tft", "peak", "peak_and_area", "peak_only", "geometry_only"]  # GCN adjacency-matrix types, same as sweep_over_options.py
METRIC_NAMES = ["Fitness", "Mo", "Pr", "Tr"]
METRIC_COLUMNS = ["fitness", "monotonicity", "prognosability", "trendability"]
FREQ_LABELS = [f"{f} kHz" for f in FREQUENCY_MAPPING] + ["average"]
GRAPH_LABELS = TYPES_LABELS
OUT_XLSX = TEST_RUN_DIR / "metrics_summary.xlsx"
OUT_DIR = TEST_RUN_DIR / "metrics_summary_results"



# ===========================================================================
# Sensor / panel geometry (moved from plot_panel.py)
# ===========================================================================
PANEL_W = 0.165  # metres. Owned by: plot_panel.py
PANEL_H = 0.240  # metres. Owned by: plot_panel.py

SENSOR_POSITIONS = np.array([
    [0.025,  0.025],   # S1
    [0.100,  0.025],   # S2
    [0.140,  0.215],   # S3
    [0.065,  0.215],   # S4
    [0.025,  0.120],   # S5
    [0.140,  0.120],   # S6
    [0.0825, 0.080],   # S7
    [0.0825, 0.160],   # S8
])  # Owned by: plot_panel.py. Imported by: GCN_train.py, imagining_alghoritm.py, visualize_crossing_graph.py, weight_matrix.py, heatmap_AE.py, make_animations.py

SENSOR_PAIRS = [(i, j) for i, j in combinations(range(1, 9), 2)]  # 28 unique 1-indexed sensor pairs. Owned by: plot_panel.py

DAMAGE_POINTS = {
    103: np.array([0.050,  0.080]),
    104: np.array([0.025,  0.080]),
    105: np.array([0.115,  0.160]),  # (165-50, 240-80) mm
    109: np.array([0.0825, 0.140]),  # (165-82.5, 240-100) mm
    123: np.array([[0.0825 - 0.03, 0.045], [0.0825, 0.045], [0.0825 - 0.03, 0.045 + 0.03], [0.0825, 0.045 + 0.03]]),  # square debond surface
}  # Owned by: plot_panel.py

# ===========================================================================
# Physical / WCPDI algorithm constants (moved from imagining_alghoritm.py)
# ===========================================================================
WAVE_VELOCITY = 1560         # v. Owned by: imagining_alghoritm.py
TIME_CONST = 0.001   # T. Owned by: imagining_alghoritm.py
MAX_DIST = WAVE_VELOCITY * TIME_CONST  # Imported by: GCN_train.py

BETA_CONSTANT = 50         # Repeated as a bare literal in: imagining_alghoritm.py, GCN_train.py, make_animations.py
DEFAULT_N_PIXELS = 1000000   # Repeated as a bare literal in: GCN_train.py, make_animations.py

# ===========================================================================
# Model hyperparameter defaults
# ===========================================================================
DEFAULT_K_SPARSE = 10          # Repeated independently 11x across fc_AE.py and big_train.py
DEFAULT_N_FEATURES = 66        # Duplicated as N_FEAT in big_train.py and the n_feat default in fc_AE.py
DEFAULT_GCN_FEATURES = 24      # Duplicated as N_GCN_FEAT in big_train.py and the gcn_feat default in fc_AE.py
DEFAULT_BATCH_SIZE = 15             # Duplicated as BATCH_SIZE in big_train.py and the batch_size default in GCN_train.py
K_SPARSE_PENALTY_WEIGHT = 0.01  # Owned by: BO.py
EPOCHS_PER_FOLD_AE = 50
EPOCHS_PER_FOLD_GCN = 200
LR = 0.001  # Learning rate for Adam optimizer. Used by: , BO_AE.py
MAX_TRIALS_AE = 30
MAX_TRIALS_GCN = 30
CNN_FIXED_LATENT_DIM = 24
DAMAGE_LOSS_WEIGHT = 0.01  # small fixed weight; damage_map_loss is unbounded, fitness is ~0-3 -- check magnitudes and adjust. Used by: BO.py, GCN_train.py
ENABLE_DAMAGE_LOSS = True  # Whether to include the damage_map_loss term in the GCN's loss function. Used by: BO.py, GCN_train.py
ENABLE_PATH_LOSS = True
ENABLE_GLOBAL_LOSS = True  # Whether to include the global_loss term in the GCN's loss function. Used by: BO.py, GCN_train.py
# ===========================================================================
# Colormaps
# ===========================================================================
CMAP_HEATMAP = "hot"          # heatmap_AE.py, imagining_alghoritm.py
CMAP_SEQUENTIAL = "viridis"   # damage_map.py, weight_matrix.py, visualize_crossing_graph.py
CMAP_DIVERGING = "plasma"     # plot_panel.py, damage_map.py
CMAP_BLUES = "Blues"          # visualize_crossing_graph.py, weight_matrix.py

palette_rgb = [
    (21, 96, 130), (233, 113, 50), (166, 202, 236),   # dark teal-blue, orange, light blue (original)
    (10, 61, 82), (166, 78, 31),                       # navy, burnt orange (darker shades)
    (120, 168, 178), (247, 178, 120),                  # steel teal, peach (mid/light tints)
    (90, 130, 90),                                     # muted sage accent
]
CUSTOM_PALETTE = [(r / 255, g / 255, b / 255) for r, g, b in palette_rgb]

