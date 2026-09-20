'''
This file opens 6 folders with BO results for every path

Creates an array of lists that is structured in the following way:
fold x frequency x panel x path x sHI
where
fold axis has size 5 (4 folds + 1 ensemble)
frequency axis has size 6
panel axis has size 5
path axis has size 28
and sHI has different size depending on the panel number

Then metrics array is created with the following structure:
frequency x path x metrics
metrics has size 4 (Fitness, Mo, Pr, Tr) and frequency has size 7 (6 frequencies + 1 average over frequencies)
Metrics are computed jointly across all 5 panels (monotonicity/trendability/prognosability
don't decompose per panel) using only the ensemble model's row of the sHI array -- not the
4 individual leave-one-out fold models.

Following plots are generated:
- sHI_grid.svg - 5x5 grid of HI vs life fraction, one column per fold, one column per frequency, lines are panels  
- WCPDI_AE_damage_maps_grid.svg  - 5x5 grid with damage maps, columns are life fractions, rows are panels
- fitness, mo, pr,, tr per path plots 
'''

import gc
import pickle
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
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ae_cross_validation_helper import ClipLayer, _CompatHeNormal, _normalization_mean_variance_axis, _predict_dataset
from CNN_AE import KSparse, ExpandLastDim, SqueezeLastDim
from AE_train import monotonicity_loss
from create_datastores import prepare_datastores
from prognostic_criteria import monotonicity_criterion, trendability_criterion, prognosability_criterion
from config import BASE_PANELS, PROJECT_ROOT, TEST_PANEL, TEST_RUN_DIR, DEFAULT_N_PIXELS, FREQUENCY_MAPPING, LIFETIME_FRACTIONS, METRIC_NAMES, FOLD_KEYS, AE_RESULTS_DIR


BO_folders = [f"Multi_path_BO_fixed_freq{i}" for i in range(6)]
PANELS = BASE_PANELS + TEST_PANEL        # panel axis (size 5), test panel always last
N_PATHS = 28
OUT_DIR = AE_RESULTS_DIR


CUSTOM_OBJECTS = {
    "KSparse": KSparse,
    "ExpandLastDim": ExpandLastDim,
    "SqueezeLastDim": SqueezeLastDim,
    "ClipLayer": ClipLayer,
    "monotonicity_loss": monotonicity_loss,
    "HeNormal": _CompatHeNormal,
}


def _collect_ensemble_DI(ensemble_model, ds_dict, norm_stats, panels):
    '''sHI predictions per panel from the ensemble model, in state order.

    The ensemble re-normalizes raw input per branch internally (see build_ensemble_ae), so
    ds_dict's already-normalized entries (normalized with `norm_stats`, see
    states_check.prepare_datastores) are first denormalized back to raw units.'''
    input_shape = ensemble_model.input_shape[1:]
    mean_b, variance_b, _ = _normalization_mean_variance_axis(norm_stats, input_shape)
    std_b = np.sqrt(variance_b)
    shi_idx = ensemble_model.output_names.index("sHI")

    DI = []
    for panel in panels:
        panel_shi = []
        for x_norm, _ in ds_dict[panel]:
            x_norm = np.asarray(x_norm).reshape((-1,) + input_shape)
            x_raw = x_norm * std_b + mean_b
            pred = ensemble_model.predict(x_raw, verbose=0)
            panel_shi.append(np.asarray(pred[shi_idx]).reshape(-1))
        DI.append(np.concatenate(panel_shi))
    return DI


def compute_sHI_and_metrics():
    '''Builds the sHI array (fold x frequency x panel x path -> variable-length array) and
    the metrics array (frequency x path x metric, ensemble only).'''
    sHI = [[[[None] * N_PATHS for _ in PANELS] for _ in BO_folders] for _ in FOLD_KEYS]
    metrics = np.full((len(BO_folders) + 1, N_PATHS, len(METRIC_NAMES)), np.nan)

    for freq_idx, folder in enumerate(BO_folders):
        freq_i = freq_idx
        for path_i in range(N_PATHS):
            path_dir = TEST_RUN_DIR / folder / f"Bayesian_CNN_AE_path{path_i}"
            if not path_dir.exists():
                print(f"[{folder}] path {path_i}: not trained yet, skipping")
                continue

            print(f"[{folder}] path {path_i}: loading models")
            ref_ds_dict, ref_norm_stats = None, None
            try:
                # 4 leave-one-out fold models -- each needs datastores normalized with
                # that same fold's own training stats (the 3 other base panels).
                for fold_idx, held_out in enumerate(BASE_PANELS):
                    model_path = path_dir / f"Model_val_{held_out}.keras"
                    if not model_path.exists():
                        print(f"  Model_val_{held_out}.keras missing, skipping fold")
                        continue

                    tf.keras.backend.clear_session()
                    fold_model = tf.keras.models.load_model(
                        str(model_path), custom_objects=CUSTOM_OBJECTS, compile=False,
                    )
                    train_ds_names = [p for p in BASE_PANELS if p != held_out]
                    _, _, _, ds_dict, *_rest = prepare_datastores(
                        path_i=path_i, freq_i=freq_i, base_batch_size=16, test_batch_size=1,
                        train_ds_names=train_ds_names, val_ds_names=[held_out], test_ds_names=TEST_PANEL,
                        include_benchmark=True, 
                    )
                    norm_stats = _rest[-1]
                    if ref_ds_dict is None:
                        # Reused below to recover raw features for the ensemble -- any
                        # fold's own (ds_dict, norm_stats) pair works, they must just match.
                        ref_ds_dict, ref_norm_stats = ds_dict, norm_stats

                    for panel_idx, panel in enumerate(PANELS):
                        panel_shi, _ = _predict_dataset(fold_model, ds_dict[panel])
                        sHI[fold_idx][freq_idx][panel_idx][path_i] = panel_shi.reshape(-1)

                    del fold_model
                    gc.collect()

                # Ensemble model -- last fold slot.
                ensemble_path = path_dir / "ensemble_model.keras"
                if ensemble_path.exists() and ref_ds_dict is not None:
                    tf.keras.backend.clear_session()
                    ensemble_model = tf.keras.models.load_model(
                        str(ensemble_path), custom_objects=CUSTOM_OBJECTS, compile=False,
                    )
                    DI = _collect_ensemble_DI(ensemble_model, ref_ds_dict, ref_norm_stats, PANELS)
                    ensemble_fold_idx = len(FOLD_KEYS) - 1
                    for panel_idx, panel_di in enumerate(DI):
                        sHI[ensemble_fold_idx][freq_idx][panel_idx][path_i] = panel_di

                    Mo = monotonicity_criterion(DI)
                    Tr = trendability_criterion(DI)
                    Pr = prognosability_criterion(DI)
                    metrics[freq_idx, path_i] = [Mo + Tr + Pr, Mo, Pr, Tr]

                    del ensemble_model
                else:
                    print(f"  ensemble_model.keras missing, skipping metrics")
            except Exception as e:
                print(f"[{folder}] path {path_i}: FAILED ({e!r})")
            finally:
                gc.collect()

    # Average over the 6 raw frequencies (ignoring NaN for missing path/freq combos).
    metrics[len(BO_folders)] = np.nanmean(metrics[:len(BO_folders)], axis=0)

    HI, HI_metrics = evaluate_HI_metrics(sHI)
    return sHI, metrics, HI_metrics, HI


def average_sHI_over_frequency(sHI, panel, fold="ensemble"):
    '''Per-path sHI curve for one panel/fold, averaged over the 6 raw frequencies
     Returns a length-N_PATHS list of 1-D arrays (state order)'''
    fold_idx = FOLD_KEYS.index(fold)
    panel_idx = PANELS.index(panel)
    averaged = []
    for path_i in range(N_PATHS):
        curves = [sHI[fold_idx][freq_idx][panel_idx][path_i] for freq_idx in range(len(BO_folders))]
        curves = [c for c in curves if c is not None]
        averaged.append(np.mean(np.stack(curves, axis=0), axis=0) if curves else None)
    return averaged

def evaluate_HI_metrics(sHI):
    '''Computes HI by averaging over the 28 paths (per fold/frequency/panel),
       then computes 4 metrics (Fitness, Mo, Pr, Tr) for every frequency and the average over frequencies, for every fold. 
       Returns (HI, HI_metrics) -- HI has fold x frequency x panel -> 1-D array structure'''

    HI = [[[None] * len(PANELS) for _ in BO_folders] for _ in FOLD_KEYS]
    for fold_idx in range(len(FOLD_KEYS)):
        for freq_idx in range(len(BO_folders)):
            for panel_idx in range(len(PANELS)):
                curves = [c for c in sHI[fold_idx][freq_idx][panel_idx] if c is not None]
                if curves:
                    HI[fold_idx][freq_idx][panel_idx] = np.mean(np.stack(curves, axis=0), axis=0)

    HI_metrics = np.full((len(FOLD_KEYS), len(BO_folders) + 1, len(METRIC_NAMES)), np.nan)  # fold x frequency x metric

    for fold_idx, fold_key in enumerate(FOLD_KEYS):
        for freq_idx in range(len(BO_folders)):
            HI_f_f = [c for c in HI[fold_idx][freq_idx] if c is not None]
            if not HI_f_f:
                continue
            Mo = monotonicity_criterion(HI_f_f)
            Tr = trendability_criterion(HI_f_f)
            Pr = prognosability_criterion(HI_f_f)
            HI_metrics[fold_idx][freq_idx] = [Mo + Tr + Pr, Mo, Pr, Tr]

        # Average-over-frequency row: per panel, average that panel's curve over

        avg_curves = []
        for panel_idx in range(len(PANELS)):
            curves = [HI[fold_idx][freq_idx][panel_idx] for freq_idx in range(len(BO_folders))]
            curves = [c for c in curves if c is not None]
            if curves:
                avg_curves.append(np.mean(np.stack(curves, axis=0), axis=0))
        if avg_curves:
            Mo = monotonicity_criterion(avg_curves)
            Tr = trendability_criterion(avg_curves)
            Pr = prognosability_criterion(avg_curves)
            HI_metrics[fold_idx][len(BO_folders)] = [Mo + Tr + Pr, Mo, Pr, Tr]

    return HI, HI_metrics


def plot_metrics(metrics, out_dir=OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = np.arange(N_PATHS)
    freq_labels = [f"freq {i + 1}" for i in range(len(BO_folders))] + ["average"]

    for metric_idx, metric_name in enumerate(METRIC_NAMES):
        fig, ax = plt.subplots(figsize=(12, 6))
        for freq_idx, label in enumerate(freq_labels):
            style = "k--" if label == "average" else "-"
            ax.plot(paths, metrics[freq_idx, :, metric_idx], style, marker="o", markersize=3, label=label)
        ax.set_xlabel("Path number")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} vs Path (ensemble model)")
        ax.set_xticks(paths)
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric_name}_vs_path.svg")
        plt.close(fig)
        print(f"Saved: {out_dir / f'{metric_name}_vs_path.svg'}")

KEY_TO_TITLES = {"103": "L1-03", "104": "L1-04", "105": "L1-05", "109": "L1-09", "123": "L1-23"}

def plot_sHI_grid(sHI, out_dir=OUT_DIR, weights=None):
    '''One figure, 7x5 grid of subplots: rows are frequencies (6 raw, labeled with their
    actual value from config.FREQUENCY_MAPPING, + average over frequencies), columns are
    folds (4 leave-one-out folds + ensemble). Each subplot (i, j) has one line per panel,
    that panel's sHI curve averaged over every path with data, plotted against life
    fraction (state index normalized to [0, 1]).'''
    out_dir.mkdir(parents=True, exist_ok=True)
    freq_labels = [f"{FREQUENCY_MAPPING[i]} kHz" for i in range(len(BO_folders))] + ["WAE"]

    fig, axes = plt.subplots(len(freq_labels), len(FOLD_KEYS), figsize=(4 * len(FOLD_KEYS), 3 * len(freq_labels)), sharex=True)

    # per_freq_means[fold_idx][panel_idx] accumulates that fold/panel's per-frequency mean
    # curves, to average into the "average over frequencies" row.
    per_freq_means = [[[] for _ in PANELS] for _ in FOLD_KEYS]

    if weights is None:
        weights = np.ones((len(FOLD_KEYS), len(freq_labels)-1)) / (len(freq_labels)-1)  


    for freq_idx in range(len(BO_folders)):
        for fold_idx, fold_key in enumerate(FOLD_KEYS):
            ax = axes[freq_idx, fold_idx]
            any_data = False
            WAE_weight = weights[fold_idx][freq_idx]
            for panel_idx, panel in enumerate(PANELS):
                curves = [c for c in sHI[fold_idx][freq_idx][panel_idx] if c is not None]
                if not curves:
                    continue
                mean_curve = np.mean(np.stack(curves, axis=0), axis=0)
                life_fraction = np.linspace(0, 1, mean_curve.shape[0])
                color, linestyle = _palette_style(panel_idx)
                ax.plot(life_fraction, mean_curve, color=color, linestyle=linestyle, label=KEY_TO_TITLES.get(panel, panel))
                per_freq_means[fold_idx][panel_idx].append(WAE_weight * mean_curve)
                any_data = True
            if not any_data:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=7)
            if freq_idx == 0:
                ax.set_title(KEY_TO_TITLES.get(fold_key, fold_key), fontsize=15)
            if fold_idx == 0:
                ax.set_ylabel(freq_labels[freq_idx], fontsize=15)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis='both', labelsize=15)
            ax.grid(True)

    # WAE row: for each fold, average that fold's per-panel curves over the 6 raw frequencies,
    avg_row = len(BO_folders)
    for fold_idx, fold_key in enumerate(FOLD_KEYS):
        ax = axes[avg_row, fold_idx]
        any_data = False
        for panel_idx, panel in enumerate(PANELS):
            if not per_freq_means[fold_idx][panel_idx]:
                continue
            overall_mean = np.sum(np.stack(per_freq_means[fold_idx][panel_idx], axis=0), axis=0)
            life_fraction = np.linspace(0, 1, overall_mean.shape[0])
            color, linestyle = _palette_style(panel_idx)
            ax.plot(life_fraction, overall_mean, color=color, linestyle=linestyle, label=KEY_TO_TITLES.get(panel, panel))
            any_data = True
        if not any_data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        if fold_idx == 0:
            ax.set_ylabel(freq_labels[-1], fontsize=15)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis='both', labelsize=15)
        ax.grid(True)

    for ax in axes[-1, :]:
        ax.set_xlabel("Life fraction", fontsize=15)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=15, loc="lower center",
               bbox_to_anchor=(0.5, 0.01), ncol=len(PANELS))

    #fig.suptitle("sHI vs life fraction, averaged over paths (one line per panel)")
    fig.tight_layout(rect=[0, 0.025, 1, 1])
    save_path = out_dir / "sHI_grid.svg"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def load_cached(out_dir=OUT_DIR):
    '''Loads a previously saved (sHI, metrics, HI_metrics, HI) quadruple from out_dir, or
    returns None if any file is missing (e.g. first run).'''
    sHI_path = out_dir / "sHI.pkl"
    metrics_path = out_dir / "metrics.npy"
    HI_metrics_path = out_dir / "HI_metrics.pkl"
    HI_path = out_dir / "HI.pkl"
    if not sHI_path.exists() or not metrics_path.exists() or not HI_metrics_path.exists() or not HI_path.exists():
        return None
    with open(sHI_path, "rb") as f:
        sHI = pickle.load(f)
    metrics = np.load(metrics_path)
    with open(HI_metrics_path, "rb") as f:
        HI_metrics = pickle.load(f)
    with open(HI_path, "rb") as f:
        HI = pickle.load(f)
    return sHI, metrics, HI_metrics, HI

from graph_performance import save_HI_metrics_xlsx, _palette_style

def main(out_dir, recompute=True):
    from AE_damage_map_grid import plot_damage_map_grid

    cached = None if recompute else load_cached(out_dir=out_dir)
    if cached is not None:
        print(f"Loaded cached results from {out_dir}")
        sHI, metrics, HI_metrics, HI = cached
    else:
        sHI, metrics, HI_metrics, HI = compute_sHI_and_metrics()

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "sHI.pkl", "wb") as f:
            pickle.dump(sHI, f)
        print(f"Saved: {out_dir / 'sHI.pkl'}")

        with open(out_dir/ "HI.pkl", "wb") as f:
            pickle.dump(HI, f)
        print(f"Saved: {out_dir / 'HI.pkl'}")

        np.save(out_dir / "metrics.npy", metrics)
        print(f"Saved: {out_dir / 'metrics.npy'}")

        np.save(out_dir / "HI_metrics.npy", HI_metrics)
        with open(out_dir / "HI_metrics.pkl", "wb") as f:
            pickle.dump(HI_metrics, f)
        print(f"Saved: {out_dir / 'HI_metrics.pkl'}")

        save_HI_metrics_xlsx(HI_metrics, out_dir=out_dir, folders=BO_folders)

    fitness = HI_metrics[:, :, 0]

    WAE_weights = fitness[:, :-1] / np.nansum(fitness[:, :-1], axis=1, keepdims=True)
    plot_metrics(metrics, out_dir=out_dir)
    plot_sHI_grid(sHI,out_dir=out_dir, weights=WAE_weights)
    plot_damage_map_grid(panel_numbers=[int(p) for p in PANELS], fractions=LIFETIME_FRACTIONS,  n_pixels=DEFAULT_N_PIXELS, sHI=sHI, save_path=str(out_dir / "WCPDI_AE_damage_maps_grid.svg"))


if __name__ == "__main__":
    remaining_path_per = ["105_wo123"]
    for panel in remaining_path_per:
        out_dir = PROJECT_ROOT / f"test_{panel}" /"path_performance_results"
        main(out_dir)
    #main(recompute="--recompute" in sys.argv)
