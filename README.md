# Multitask CAE-GCN for HI extraction and damage detection 

Structural health monitoring pipeline for detecting damage and extracting HI for composite panels: raw GW signal data is turned
into per-path health indices (sHI) via autoencoders and/or a graph convolutional
network (GCN), combined using weight compensated probabilistic damage imagining (WCPDI) into a damage map on the panel, and scored with
prognostic-criteria metrics (monotonicity, trendability, prognosability).

`config.py` Central configuration: paths, panel/sensor constants, shared plot palette.

## Running the workflow

The held-out test panel is chosen with the `SHM_TEST_PANEL` environment variable (default `103`, see
`config.py`). Everything that depends on it is written under `test_<panel>_wo123/` (`TEST_RUN_DIR`), so
runs for different test panels are computed separately.

- `run_cross_test.py` (project root) is the entry point for regenerating the results. Set its `PANELS` list
  (and optional `START_FREQ` resume points) and it launches `training/run_single_panel_serial.py` in a
  fresh process for each test panel.
- `training/run_single_panel_serial.py <panel> [start_freq]` runs the whole per-panel pipeline, in order:
  AE Bayesian optimization (`BO_AE.py`) → AE performance (`path_performance.py`) → AE sHI extraction
  (`extract_shi.py`) → CAE-GCN sweep → raw-feature extraction (`features_extractor.py`) → raw-feature GCN
  sweep → test-panel fitness metrics → WAE metrics → metrics summary.
- Once every panel has run, `results_analysis/performance_accross_tests.py` (summary across test panels) and
  `results_analysis/L1_23_performance_evaluation.py` (evaluation on the unseen L1-23 damage) produce the
  cross-panel figures and tables.

## Source folders

### `tools/`
Core, reusable SHM building blocks: sensor-network graph construction, the WCPDI
damage-imaging algorithm, sHI extraction from trained autoencoders, prognostic-criteria
metrics, and the shared panel-drawing primitive other plots build on.

- `weight_matrix.py` — Builds path-to-path attention/adjacency matrices (by crossing angle, area overlap, or raw signal features) used as GCN graph edges, and detects failed sensors per panel state.
- `imagining_alghoritm.py` — Implements the WCPDI probability-based damage-imaging algorithm (`P`, `U`, `WCPDI` functions) that turns per-path sHI values into a 2D damage probability map on the panel.
- `prognostic_criteria.py` — Defines the monotonicity, trendability, and prognosability criterion functions used to score health-index quality.
- `plot_panel.py` — Defines `plot_panel_with_paths`/`_draw_static_panel` to draw the PZT panel with sensors, damage point, and optionally active sensor paths; the shared backdrop other plotting scripts draw on top of.

### `data/`
Loads raw `.mat` sensor signal data (`States_<panel>.mat`) and turns it into the
TensorFlow/PyTorch-Geometric datasets (feature vectors and path-graphs) consumed by the
AE and GCN models.

- `states.py` — Defines the `states` class that loads panel `.mat` files and exposes amplitude, benchmark amplitude, time, and energy accessors plus plotting/summary helpers.
- `features_extractor.py` — Extracts 19 time-domain + 14 frequency-domain features per signal half (66 features/state) from panel states, as input for the feature-based autoencoder.
- `create_datastores.py` — Builds TensorFlow train/val/test datasets  from the features data for autoencoder training.
- `extract_shi.py` — Extracts sHI (health-index) values and latent reconstructions from trained autoencoder models for a given panel/frequency and writes them out for use in the graph dataset.
- `graph_dataset.py` — Defines `Panel_GraphDataset`, a PyTorch-Geometric `InMemoryDataset` that turns per-path sHI/latent values and attention-based adjacency into per-state graphs for the GCN.

### `models/`
Defines the two neural-network architectures used in the pipeline: a sparse
fully-connected/CNN autoencoder for per-path health indices, and a graph convolutional
network over the sensor-path graph.

- `CNN_AE.py` — Defines custom Keras layers (`KSparse`, `ExpandLastDim`, `SqueezeLastDim`) and builders for the fully-connected and CNN sparse autoencoder architectures.
- `GCN.py` — Defines `DeepGraphCNN`, a residual multi-layer `GCNConv`-based graph neural network that maps a path graph to a scalar health index.

### `training/`
Training loops and Bayesian hyperparameter-optimization drivers for both the
autoencoder and GCN models, plus cross-validation and sensitivity-study helpers.

- `ae_cross_validation_helper.py` — Provides plotting, ensembling, and Keras-compatibility helper layers/functions (`ClipLayer`, `_predict_dataset`, ensemble builders) shared by `BO_AE.py` and `AE_train.py`'s cross-validation loop.
- `AE_train.py` — Defines the leave-one-out cross-validation training process for the fc_AE/CNN_AE autoencoders across panels 103/104/105/109.
- `BO_AE.py` — Runs Bayesian hyperparameter optimization for the fc_AE and CNN_AE autoencoders trained on time-frequency features, optionally comparing both architectures side by side.
- `BO_GCN.py` — Runs Bayesian hyperparameter optimization for `DeepGraphCNN` via leave-one-out cross-validation, then retrains and saves an ensemble model with diagnostic plots.
- `GCN_train.py` — Defines the GCN training loop, including the monotonicity loss, model/dataset setup, and a `plot_HI` visualization of learned health index vs. state.
- `sensitivity_study.py` — Sweeps `BO_GCN.py`'s Bayesian optimization across frequencies for the CAE-GCN (`run_pre_processed_sweep`), raw-feature GCN (`run_raw_sweep`) and the other adjacency/loss GCN types (`run_types_sweep`), each followed by `graph_performance.py` on the results.
- `run_single_panel_serial.py` — Worker that runs the full per-panel pipeline (AE, CAE-GCN, GCN, metrics) for one test panel given on the command line; launched once per panel by `run_cross_test.py`.

### `results_analysis/`
Standalone analysis/plotting entry points that load cached training/BO results and
produce the summary figures and tables (hyperparameters, fitness metrics, damage maps,
sHI curves).

- `AE_hyperparameters_summary.py` — Reads `BO_AE.py`'s per-frequency `best_params.json` (path 0's Bayesian search) for every test panel and plots each AE hyperparameter vs. frequency, one series per test panel.
- `GCN_hyperparameters_summary.py` — Reads `BO_GCN.py`'s `best_hyperparameters_freq{N}_pre_processed.json` for every test panel and plots each GCN hyperparameter (hidden channels, hidden dim, dropout, batch size, learning rate) vs. frequency, one series per test panel.
- `AE_damage_map_grid.py` — Generates a heatmap of the WCPDI damage map on the panel using autoencoder-derived sHI values.
- `Fitness_summary.py` — Averages the frequency x path x metric arrays from `graph_performance.py` and `path_performance.py` across paths, writing a model-type/frequency fitness table and plots.
- `Compute_WAE.py` — Computes the Weighted Average Ensemble health index and its prognostic metrics across model directories and saves/appends the results.
- `Fitness_test_metrics.py` — Computes prognostic-criteria metrics restricted to the held-out test panel for each model directory's cached HI data.
- `graph_performance.py` — Runs the full GCN-based HI/damage-map/prognostic-metric analysis (sHI grid, WCPDI damage maps, per-path metric plots) over the per-frequency GCN Bayesian-optimization results.
- `path_performance.py` — Aggregates the six per-path-per-frequency AE Bayesian-optimization folders into fold x frequency x panel x path sHI arrays, computes prognostic metrics, and plots sHI/damage-map grids.
- `performance_accross_tests.py` — Reads the cached results of all four test-panel runs (`HI.pkl`, metrics, WAE, damage maps, sHI) and produces the cross-test-panel summary: HI grids, HI-metrics xlsx, damage-map grids, fitness vs. frequency, and WAE fitness vs. GCN type. No model is reloaded.
- `L1_23_performance_evaluation.py` — Evaluates each leave-one-out test-panel model (CAE, GCN, CAE-GCN) on the unseen-damage panel 123 (L1-23): WAE HI curves, prognostic-criteria test metrics, and WCPDI damage-map grids. Intermediate results are cached in `L1_23_results_analysis/`.

### `intermediate_results_check/`
Ad-hoc diagnostic/inspection scripts for sanity-checking data and models mid-pipeline —
not part of the main result-generating flow above.

- `plot_raw_signal.py` — Loads several panels' `states` and plots raw/benchmark signal amplitude for chosen states, frequency, and path.
- `plot_sHI.py` — Loads a saved autoencoder cross-validation ensemble model and plots its sHI predictions for one panel.
- `inspect_connections.py` — Visualizes path/graph connectivity: the geometric connection matrix, weighted adjacency, subgraphs, path pairs, and panel schematic.
- `inspect_field_of_influance.py` — Plots the elliptical "field of influence" of one or more sensor paths based on `imagining_alghoritm.py`'s weighting function `U`.
- `inspect_latent_zeros.py` — Reports, per latent position, how many entries are exactly zero and the value range/mean/std across the cached AE-latent files, as a sanity check of the K-sparse latent code.
- `inspect_raw_feature_stats.py` — Same statistics as `inspect_latent_zeros.py` but for the raw features, to check the latent-space extraction against them.
- `inspect_sHI_stats.py` — Same accumulation as `inspect_latent_zeros.py`, keyed by path/state, on the cached `shi` arrays.
- `plot_envelope_peak_area.py` — Plots one signal's Hilbert envelope with the peak lobe shaded, as a cross-check of `states.py`'s `signal_envelope_peak_area`.

## Output / generated folders

These are all produced by running the scripts above — none are checked in as source,
and most can be regenerated by rerunning the corresponding script.

### Per-test-panel folders (inside `test_<panel>_wo123/`)

| Folder / file | Contents |
|---|---|
| `Multi_path_BO_fixed_freq0` … `freq5` | Per-frequency AE Bayesian-optimization output from `BO_AE.py`: one `Bayesian_CNN_AE_path{0..27}/` subfolder per path, each with fold models, an ensemble model, and a `model_database.xlsx`. Each frequency folder also holds a `*_best_params.json` with the hyperparameters found by the search on path 0, shared by all paths. |
| `results` | GCN Bayesian-optimization output from `BO_GCN.py`: `Bayesian_GCN_{peak,raw,<type>}_freq{N}/` subfolders (one per model type and frequency) plus `best_hyperparameters_freq{N}_*.json`. |
| `graph_performance_results_<type>` | `graph_performance.py`'s cached HI/metrics/damage-map results, one folder per GCN type (`basic`, `peak`, `raw`, `geometry_only`, `peak_only`, `peak_and_area`, `peak_fft`, `peak_tff`, `peak_tft`). |
| `path_performance_results` | `path_performance.py`'s cached sHI/metrics/damage-map results for the per-path AE ensemble. |
| `metrics_summary_results`, `metrics_summary.xlsx` | `Fitness_summary.py`'s cross-model-type fitness comparison plots/table. |
| `graph_data` | PyTorch-Geometric dataset root (raw/processed), written and read by `graph_dataset.py`, `GCN_train.py`, `imagining_alghoritm.py`, and `extract_shi.py`. |
| `model_database_features.xlsx` | Appended to by `AE_train.py`'s `model_train_features`; a running log of every trained AE model's hyperparameters and results. |
| `tuner_dir` | keras-tuner scratch directory used during `BO_AE.py`'s search. |

### Shared / cross-panel folders (project root)

| Folder | Contents |
|---|---|
| `features_cache` | Cached per-panel extracted features, written by `features_extractor.py`; independent of the test panel, so shared by every run. |
| `test_panel_performance_results` | `performance_accross_tests.py`'s cross-test-panel figures and tables. |
| `L1_23_results_analysis` | `L1_23_performance_evaluation.py`'s plots, xlsx table, and per-panel result caches. |
| `AE_hyperparameters_summary_results` | `AE_hyperparameters_summary.py`'s hyperparameter-vs-frequency plots. |
| `GCN_hyperparameters_summary_results` | `GCN_hyperparameters_summary.py`'s hyperparameter-vs-frequency plots. |
| `__pycache__` | Python bytecode cache — not real content, safe to delete anytime. |


