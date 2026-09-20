"""
Bayesian hyperparameter optimization for fc_AE and CNN_AE autoencoders trained on time-frequency
features 

Configure MODE / MODEL_TYPE / MAX_TRIALS at the bottom of the file:
    MODE = "single"  -> optimize one model type (uses MODEL_TYPE)
    MODE = "duo"     -> optimize both fc_AE and CNN_AE, plot side-by-side

Hyperparameters for FC_AE:
    - latent_dim, k_sparse (bounded by latent_dim), drop_rate, batch_size

Hyperparameters for CNN_AE:
    - k_sparse (bounded by latent_dim), filters_bench, filters_path, batch_size
    - latent_dim is fixed at CNN_FIXED_LATENT_DIM (not tuned) for this architecture

Hardcoded constants:
    - K_SPARSE_PENALTY_WEIGHT: weight for the k-sparse penalty subtracted from the search
      objective (see below) -- larger k_sparse is penalized so the search doesn't just
      chase whatever k_sparse happens to maximize raw fitness.
    - CV_PANELS: panels used for cross-validation (leave-one-out)
    - EPOCHS_PER_FOLD: number of epochs to train in each fold of cross-validation
    - MODEL_DB_DIR: directory to save the model database Excel file
    - learning_rate: learning rate for training the models (currently fixed at 0.001)
    - loss_weights: weights for the reconstruction loss and latent loss (currently fixed at 1.0 and 2.0)
    - Test set is config.TEST_PANEL

Search objective: "Objective" (direction="max") = mean_fitness - k_sparse_penalty(k_sparse),
where mean_fitness is computed per CV fold as a*monotonicity + b*trendability +
c*prognosability (see fitness_objective, and prognostic_criteria.py for the three
underlying criteria) from each fold's trained model's own sHI predictions across
CV_PANELS, and k_sparse_penalty is a fixed per-trial penalty (weight * k_sparse) that
discourages the search from picking larger k_sparse values purely because they raise
raw fitness. mean_fitness itself is also reported every trial as an unpenalized
diagnostic (that's what's plotted). mean_val_loss / mean_train_loss are likewise still
computed and returned every trial purely as training diagnostics (plotted, but don't
drive the search) -- note their magnitude isn't directly comparable across different
batch_size values or training epochs, since monotonicity_loss sums (not averages) an
unbounded per-pair term over the batch, so it scales with batch_size and with how
monotonic the current predictions happen to be, independent of overall model quality.

At the end train and validation datasets are assumed and latent representations and signal reconstructions are ploted.
"""


import gc
import json
import multiprocessing
import os
import sys
from functools import partial
from pathlib import Path
import psutil, os

# for checking memory usage during optimization
def log_mem(tag=""):
    rss = psutil.Process(os.getpid()).memory_info().rss
    print(f"[MEM {tag}] {rss / 1e9:.2f} GB")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "tools", "training", "intermediate_results_check", "results_analysis"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from AE_train import model_train_features, monotonicity_loss
from CNN_AE import (
    build_CNN_AE_features,
    build_fc_AE_features,
)
from ae_cross_validation_helper import (
    plot_sHI_cv_fold, plot_reconstruction_cv_fold, build_ensemble_ae, plot_ensemble_sHI,
)
from create_datastores import prepare_datastores
from prognostic_criteria import monotonicity_criterion, trendability_criterion, prognosability_criterion
from config import (
    K_SPARSE_PENALTY_WEIGHT, CV_PANELS, VAL_PANELS,
    BO_RESULTS_DIR, BO_TUNER_DIR, BO_SEARCH_RESULTS_DIR, TEST_RUN_DIR,
    DEFAULT_N_FEATURES, TEST_PANEL, EPOCHS_PER_FOLD_AE, CNN_FIXED_LATENT_DIM, LR, MAX_TRIALS_AE,
)

# ─── Entry point ──────────────────────────────────────────────────────────────
# Configure the run here:
MODE = "single"            # "single" or "duo"
MODEL_TYPE = "CNN_AE"    # used only when MODE == "single"  ("fc_AE" or "CNN_AE")


# ─── Constants ────────────────────────────────────────────────────────────────

# HP columns reported in sensitivity / coverage plots, per model type.
HP_COLS = {
    # Fully connected AE has the following HP
    "fc_AE": [
         "k_sparse_frac", "batch_size", "latent_dim", "drop_rate", "l2_reg"
    ],
    # CNN AE has the following HP
    "CNN_AE": [
        "k_sparse_frac", "filters_bench", "filters_path",  "batch_size", "l2_reg"
    ],
}

MODEL_DB_DIR = str(BO_RESULTS_DIR)


# ─── HP helpers & k-sparse penalty ────────────────────────────────────────────
def k_sparse_penalty(k, weight=K_SPARSE_PENALTY_WEIGHT):
    return float(weight) * float(k)


def get_latent_dim_hp(hp):
    return hp.Int("latent_dim", min_value=4, max_value=24, step=4)


def get_k_sparse_frac_hp(hp):
    # A fraction of latent_dim, not an absolute count
    return hp.Float("k_sparse_frac", min_value=0.1, max_value=0.9, step=0.1)


def resolve_k_sparse(k_sparse_frac, latent_dim):
    # changes the k_sparse_frac to an absolute k_sparse value, bounded by latent_dim
    return max(2, min(latent_dim - 1, round(k_sparse_frac * latent_dim)))


def get_drop_rate_hp(hp):
    return hp.Float("drop_rate", 0.0, 0.5, step=0.1)


def get_batch_size_hp(hp):
    return hp.Int("batch_size", min_value=4, max_value=32, step=4)


def get_l2_reg_hp(hp):
    return hp.Float("l2_reg", min_value=1e-5, max_value=1e-2, sampling="log")


# ─── Model builder ────────────────────────────────────────────────────────────
def build_model(hp, model_type="fc_AE"):

    # extract hyperparameters
    latent_dim = CNN_FIXED_LATENT_DIM if model_type == "CNN_AE" else get_latent_dim_hp(hp)
    k_sparse = resolve_k_sparse(get_k_sparse_frac_hp(hp), latent_dim)
    drop_rate = get_drop_rate_hp(hp) if model_type == "fc_AE" else None
    l2_reg = get_l2_reg_hp(hp)

    if model_type == "fc_AE":
        params = {
            "input_size": DEFAULT_N_FEATURES,
            "n_features": DEFAULT_N_FEATURES,
            "latent_dim": latent_dim,
            "k_sparse": k_sparse,
            "drop_rate": drop_rate,
            "l2_reg": l2_reg,
        }
        model = build_fc_AE_features(params)
    elif model_type == "CNN_AE":
        params = {
            "n_features": DEFAULT_N_FEATURES,
            "n_channels": 2,
            "latent_dim": latent_dim,
            "k_sparse": k_sparse,
            "filters_bench": hp.Int("filters_bench", min_value=6, max_value=20, step=2),
            "filters_path": hp.Int("filters_path", min_value=4, max_value=12, step=2),
            "l2_reg": l2_reg,
        }
        model = build_CNN_AE_features(params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer,
        loss={"reconstruction": "mse", "sHI": monotonicity_loss},
        loss_weights={"reconstruction": 1.0, "sHI": 2.0},
    )
    return model


# ─── Fitness objective (prognostic_criteria.py) ───────────────────────────────
def _collect_sHI(model, ds_dict, panels):
    shi_idx = model.output_names.index("sHI")
    DI = []
    for panel in panels:
        pred = model.predict(ds_dict[panel], verbose=0)
        DI.append(np.asarray(pred[shi_idx]).reshape(-1))
    return DI


def fitness_objective(model, ds_dict, panels, a=1.0, b=1.0, c=1.0):
    """Higher is better. trendability_criterion needs >=2 panels to correlate against
    each other, so `panels` should be the full CV panel set, not a single held-out one."""
    DI = _collect_sHI(model, ds_dict, panels)
    monotonicity = monotonicity_criterion(DI)
    trendability = trendability_criterion(DI)
    prognosability = prognosability_criterion(DI)
    return a * monotonicity + b * trendability + c * prognosability


# ─── Custom tuner ─────────────────────────────────────────────────────────────
class MyTuner(kt.BayesianOptimization):
    def __init__(self, *args, path_i, freq, model_type=None, 
                 cv_panels=None, epochs_per_fold=EPOCHS_PER_FOLD_AE, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = model_type
        self.benchmark = True if model_type == "CNN_AE" else False
        self.diff_bench = True if model_type == "fc_AE" else False
        self.path_i = path_i
        self.freq = freq
        self.cv_panels = cv_panels or VAL_PANELS
        self.train_pannels = CV_PANELS
        self.epochs_per_fold = epochs_per_fold

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        bs = get_batch_size_hp(hp)
        
        latent_dim = CNN_FIXED_LATENT_DIM if self.model_type == "CNN_AE" else get_latent_dim_hp(hp)
        k_sparse = resolve_k_sparse(get_k_sparse_frac_hp(hp), latent_dim)

        fold_val = []
        fold_tr = []
        fold_fitness = []

        # run cross-validation folds for every panel in the cv_panels
        for fold_idx, val_panel in enumerate(self.cv_panels):

            log_mem(f"trial {trial.trial_id} fold {fold_idx} before prep")

            train_panels = [p for p in self.train_pannels if p != val_panel]
            print(f"\n[trial {trial.trial_id}] fold {fold_idx+1}/"
                  f"{len(self.cv_panels)} — val panel {val_panel}")

            train_ds, val_ds, _, ds_dict, *_ = prepare_datastores(
                path_i=self.path_i,
                freq_i=self.freq,
                base_batch_size=bs,
                test_batch_size=1,
                train_ds_names=train_panels,
                val_ds_names=[val_panel],
                test_ds_names=TEST_PANEL,
                include_benchmark=self.benchmark,
                diff_bench=self.diff_bench,
            )

            log_mem(f"trial {trial.trial_id} fold {fold_idx} after prep")

            # Fresh, compiled model for this fold
            model = self.hypermodel.build(hp)

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.epochs_per_fold,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=10,
                        restore_best_weights=True,
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.5,
                        patience=4,
                        min_lr=1e-5,
                    ),
                ],
                verbose=0,
            )

            h = history.history
            fold_val.append(min(h.get("val_loss", [float("inf")])))
            fold_tr.append(min(h.get("loss", [float("inf")])))

            fold_fitness.append(fitness_objective(model, ds_dict, self.train_pannels))

            # release graph/memory between folds
            del model, history, train_ds, val_ds, ds_dict, h
            tf.keras.backend.clear_session()
            gc.collect()
            log_mem(f"trial {trial.trial_id} fold {fold_idx} END")

        log_mem(f"trial {trial.trial_id} END")
        
        mean_fitness = float(np.mean(fold_fitness))
        objective = mean_fitness - k_sparse_penalty(k_sparse)
        
        return {
            "Objective":        objective,
            "mean_fitness":     mean_fitness,
            "std_fitness":      float(np.std(fold_fitness)),
            "mean_val_loss":    float(np.mean(fold_val)),
            "mean_train_loss":  float(np.mean(fold_tr)),
        }


# ─── Helpers for parsing tuner results ────────────────────────────────────────
def _scalar(v):
    """Cast to float"""
    if hasattr(v, "value"):
        v = v.value
    if isinstance(v, tf.Tensor):
        v = v.numpy()
    if isinstance(v, np.ndarray):
        return float(v.reshape(-1)[0])
    if isinstance(v, (list, tuple)):
        return float(v[0])
    return float(v)


def _collect_loss_histories(tuner):
    val_h, train_h, fitness_h = [], [], []
    for t in tuner.oracle.trials.values():
        if t.score is None:  # skip incomplete trials
            continue
        val_obs     = [_scalar(o.value) for o in t.metrics.get_history("mean_val_loss")]
        train_obs   = [_scalar(o.value) for o in t.metrics.get_history("mean_train_loss")]
        fitness_obs = [_scalar(o.value) for o in t.metrics.get_history("mean_fitness")]
        if val_obs and train_obs and fitness_obs:
            val_h.append(min(val_obs))
            train_h.append(min(train_obs))
            fitness_h.append(max(fitness_obs))
    return train_h, val_h, fitness_h


def _build_trials_dataframe(tuner):
    records = []
    for tid, t in tuner.oracle.trials.items():
        if t.score is None:
            continue
        row = {"trial_id": tid, "objective": t.score}
        row.update(t.hyperparameters.values)

        val_hist       = [_scalar(o) for o in t.metrics.get_history("mean_val_loss")]
        train_hist     = [_scalar(o) for o in t.metrics.get_history("mean_train_loss")]

        row["final_val_loss"]   = val_hist[-1]   if val_hist   else None
        row["final_train_loss"] = train_hist[-1] if train_hist else None
        row["overfit_gap"] = (
            row["final_val_loss"] - row["final_train_loss"]
            if (row["final_val_loss"] is not None
                and row["final_train_loss"] is not None)
            else None
        )
        records.append(row)
    return pd.DataFrame(records).sort_values("objective", ascending=False)


def _save_best_trial_details(best, t_elapsed, out_dir):
    with open(f"{out_dir}/best_trial_details.txt", "w") as f:
        f.write(f"Optimization time {t_elapsed}\n")
        f.write(f"Trial ID: {best.trial_id}\n")
        f.write(f"Objective (mean_fitness - k_sparse_penalty): {best.score}\n")
        f.write(f"Hyperparameters: {best.hyperparameters.values}\n")
        f.write(f"Final fitness: {best.metrics.get_last_value('mean_fitness')}\n")
        f.write(f"Final val_loss: {best.metrics.get_last_value('mean_val_loss')}\n")
        f.write(f"Final train_loss: {best.metrics.get_last_value('mean_train_loss')}\n")
        f.write(f"Train loss history: {best.metrics.get_history('mean_train_loss')}\n")


def _append_to_model_database(model_info, results_dir=MODEL_DB_DIR):
    os.makedirs(results_dir, exist_ok=True)
    excel_path = os.path.join(results_dir, "model_database.xlsx")
    df_new = pd.DataFrame([model_info])

    if not os.path.exists(excel_path):
        df_new.to_excel(excel_path, index=False)
        print(f"Created new database: {excel_path}")
        return

    with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl",
                        if_sheet_exists="overlay") as writer:
        try:
            existing_df = pd.read_excel(excel_path)
            df_new.to_excel(writer, index=False, header=False,
                            startrow=len(existing_df) + 1)
        except Exception:
            df_new.to_excel(excel_path, index=False)
    print(f"Updated database at {excel_path}")

# ─── Main optimization routine ────────────────────────────────────────────────
def run_bayesian_optimization(path_i, freq, model_type="fc_AE", max_trials=MAX_TRIALS_AE, out_dir=None, db_dir=None, do_BO=True):
    """Run Bayesian search, then retrain with the best hyperparameters using the same
    leave-one-out cross-validation loop big_train.py's __main__ runs (see the retrain
    loop below) -- saving/plotting each fold plus an ensemble directly into out_dir.

    Returns a dict with:
        model_type, fold_models, ensemble_model, best_params,
        train_h, val_h, running_best, df, out_dir
    """
    if model_type == "fc_AE":
        model_building_function = build_model
    elif model_type == "CNN_AE":
        model_building_function = partial(build_model, model_type="CNN_AE")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    
    date = pd.Timestamp.now().strftime("%Y_%m_%d-%H_%M_%S")
    out_dir = str(BO_SEARCH_RESULTS_DIR / f"Bayesian_{model_type}_{date}") if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)

    if do_BO:
        t_start = pd.Timestamp.now()
        tuner = MyTuner(
            model_building_function,
            objective=kt.Objective("Objective", direction="max"),
            max_trials=max_trials,
            directory=str(BO_TUNER_DIR),
            project_name="ae",
            overwrite=True,
            model_type=model_type,
            path_i=path_i,
            freq=freq
        )
        tuner.search()
        t_elapsed = pd.Timestamp.now() - t_start
        print(f"Bayesian optimization ({model_type}) completed in {t_elapsed}")


        best = tuner.oracle.get_best_trials(1)[0]
        print("Trial ID:", best.trial_id)
        print("Objective (mean_fitness - k_sparse_penalty):", best.score)
        print("Hyperparameters:", best.hyperparameters.values)
        print("Final val_loss:", best.metrics.get_last_value("mean_val_loss"))
        print("Final train loss:", best.metrics.get_last_value("mean_train_loss"))

        _save_best_trial_details(best, t_elapsed, out_dir)

        train_h, val_h, fitness_h = _collect_loss_histories(tuner)
        running_best = np.maximum.accumulate(fitness_h) if fitness_h else np.array([])
        df = _build_trials_dataframe(tuner)
        print(df.head(10))

    # Retrain with the best hyperparameters using leave-one-out cross-validation
    params_path = TEST_RUN_DIR / f"Multi_path_BO_fixed_freq{freq}" / f"Bayesian_{model_type}_test{TEST_PANEL}_freq{freq}_best_params.json"
    if do_BO:
        best_params = best.hyperparameters.values
        # save the best hyperparameters as a dictionary for other paths
        params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"Saved: {params_path}")
    else:
        # Load the best hyperparameters found for another path at this freq/model_type
        with open(params_path, "r") as f:
            best_params = json.load(f)
    seed=42

    latent_dim = CNN_FIXED_LATENT_DIM if model_type == "CNN_AE" else best_params["latent_dim"]
    final_params = {
        "input_size": DEFAULT_N_FEATURES,
        "n_features": DEFAULT_N_FEATURES,
        "latent_dim": latent_dim,
        "k_sparse": resolve_k_sparse(best_params["k_sparse_frac"], latent_dim),
        "l2_reg": best_params["l2_reg"],
    }
    if model_type == "fc_AE":
        final_params["drop_rate"] = best_params["drop_rate"]
    if model_type == "CNN_AE":
        final_params["filters_bench"] = best_params["filters_bench"]
        final_params["filters_path"] = best_params["filters_path"]
        final_params["n_channels"] = 2  

    rec_train_loss_list, lat_train_loss_list = [], []
    rec_val_loss_list, lat_val_loss_list = [], []
    fold_final_losses = []
    fold_entries = []  # keep the mean and std for each fold to normalize the input corrctly when building the ensemble model
    last_ds_dict, last_States_dict = None, None
    model_info = None

    # train and validate on each fold, save the model and plot sHI and reconstruction for each fold
    for panel in CV_PANELS:
        train_ds_names = [p for p in CV_PANELS if p != panel]
        val_ds_names = [panel]
        tf.random.set_seed(seed)

        (model, history, final_loss, _, _, _, _, _, model_info, ds_dict, _, States_dict,
         *_rest) = model_train_features(
            net_type=model_type,
            include_benchmark=(model_type == "CNN_AE"),
            params=final_params,
            path_i=path_i,
            frequency_i=freq,
            train_ds_names=train_ds_names,
            val_ds_names=val_ds_names,
            results_dir=out_dir,
            epochs=EPOCHS_PER_FOLD_AE,
            base_batch_size=best_params["batch_size"],
            loss_weights={"reconstruction": 1.0, "sHI": 2.0},
            seed=seed,
            filepath=f"Model_val_{panel}.keras",
        )
        norm_stats = _rest[-1]

        plot_sHI_cv_fold(model, ds_dict, States_dict, CV_PANELS, panel,
                          save_path=os.path.join(out_dir, f"sHI_fold_val_{panel}.png"))
        plot_reconstruction_cv_fold(model, ds_dict, States_dict, CV_PANELS, panel, 0,
                                     save_path=os.path.join(out_dir, f"reconstruction_fold_val_{panel}.png"))

        rec_train_loss_list.append(history.history['reconstruction_loss'])
        lat_train_loss_list.append(history.history['sHI_loss'])
        rec_val_loss_list.append(history.history['val_reconstruction_loss'])
        lat_val_loss_list.append(history.history['val_sHI_loss'])
        fold_final_losses.append(final_loss)

        fold_entries.append((panel, model, norm_stats))
        last_ds_dict, last_States_dict = ds_dict, States_dict

    _append_to_model_database(model_info, results_dir=db_dir or MODEL_DB_DIR)

    # Learning curves: reconstruction loss (top row) + latent loss (bottom row), one
    # column per panel -- identical layout to big_train.py's __main__.
    avg_final_loss = float(np.mean(fold_final_losses))
    plt.figure(figsize=(16, 6))
    for i, panel in enumerate(CV_PANELS):
        plt.subplot(2, 4, i + 1)
        plt.plot(rec_train_loss_list[i], label='Train')
        plt.plot(rec_val_loss_list[i], label='Val')
        plt.title(f'Recon Loss - Panel {panel}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    for i, panel in enumerate(CV_PANELS):
        plt.subplot(2, 4, i + 5)
        plt.plot(lat_train_loss_list[i], label='Train')
        plt.plot(lat_val_loss_list[i], label='Val')
        plt.title(f'Latent Loss - Panel {panel}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"learning_curves_{model_type}_{avg_final_loss:.4f}.png"))
    plt.close()

    # Ensemble of the 4 cross-validation folds -- same as big_train.py's __main__.
    ensemble_model = build_ensemble_ae(fold_entries)
    ensemble_model.save(os.path.join(out_dir, "ensemble_model.keras"))
    _, _, ref_norm_stats = fold_entries[0]
    plot_ensemble_sHI(
        ensemble_model, last_ds_dict, ref_norm_stats, last_States_dict, CV_PANELS,
        save_path=os.path.join(out_dir, "ensemble_sHI.png"),
    )

    if do_BO:
        return {
            "model_type": model_type,
            "fold_models": fold_entries,
            "ensemble_model": ensemble_model,
            "best_params": best_params,
            "train_h": train_h,
            "val_h": val_h,
            "fitness_h": fitness_h,
            "running_best": running_best,
            "df": df,
            "out_dir": out_dir,
        }
    else:
        return {
            "model_type": model_type,
            "fold_models": fold_entries,
            "ensemble_model": ensemble_model,
            "best_params": best_params,
            "out_dir": out_dir,
        }


# ─── Plotting ─────────────
def _min_max_normalize(values):
    x = np.asarray(values, dtype=float)
    lo, hi = x.min(), x.max()
    if hi <= lo:  # all trials had the same value (e.g. a single trial) -- avoid /0
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def plot_optimization_progress(results, save_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4), squeeze=False)
    for ax, r in zip(axes[0], results):
        ax.plot(_min_max_normalize(r["fitness_h"]), label="Fitness (min-max)")
        ax.plot(_min_max_normalize(r["val_h"]), label="Validation Loss (min-max)")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Min-Max Normalized [0, 1]")
        ax.set_title(f"Fitness & Validation Loss per Trial — {r['model_type']}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()


def plot_running_best(results, save_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4), squeeze=False)
    for ax, r in zip(axes[0], results):
        ax.plot(r["running_best"], "k--", label="Running Best")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Best Fitness")
        ax.set_title(f"Running Best — {r['model_type']}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()


def plot_overfitting(results, save_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    for ax, r in zip(axes[0], results):
        df = r["df"]
        sc = ax.scatter(df["final_train_loss"], df["final_val_loss"],
                        c=df["overfit_gap"], cmap="RdYlGn_r",
                        edgecolors="k", s=60)
        fig.colorbar(sc, ax=ax, label="overfit gap (val - train)")
        lo, hi = df["final_train_loss"].min(), df["final_train_loss"].max()
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="no overfit line")
        ax.set_xlabel("Final train loss")
        ax.set_ylabel("Final val loss")
        ax.set_title(f"Overfitting — {r['model_type']}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()


def plot_hp_sensitivity(result, save_path):
    """Per-model HP scatter (HP cols differ per model, so this is single-model)."""
    df = result["df"]
    cols = HP_COLS[result["model_type"]]
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 8), squeeze=False)
    for ax, col in zip(axes.flat, cols):
        ax.scatter(df[col], df["objective"], alpha=0.6,
                   edgecolors="k", linewidths=0.3)
        ax.set_xlabel(col); ax.set_ylabel("objective")
        ax.set_title(f"{col} vs objective")
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    fig.suptitle(f"HP Sensitivity — {result['model_type']}", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()


def plot_hp_coverage(result, save_path):
    df = result["df"]
    cols = HP_COLS[result["model_type"]]
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 8), squeeze=False)
    for ax, col in zip(axes.flat, cols):
        ax.hist(df[col].dropna(), bins=10, edgecolor="k")
        ax.set_title(col)
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    fig.suptitle(f"Sampled HP Distribution — {result['model_type']}", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()


def plot_all(results):
    """Generate every diagnostic plot.

    Combined plots (progress / running_best / overfitting) go into the first
    result's out_dir. HP sensitivity / coverage are per-model, into each
    result's own out_dir.
    """
    primary_dir = results[0]["out_dir"]
    plot_optimization_progress(results, f"{primary_dir}/bayesian_optimization_progress.svg")
    plot_running_best(results,         f"{primary_dir}/bayesian_running_best.svg")
    plot_overfitting(results,          f"{primary_dir}/overfitting_analysis.svg")
    for r in results:
        plot_hp_sensitivity(r, f"{r['out_dir']}/hp_sensitivity.svg")
        plot_hp_coverage(r,    f"{r['out_dir']}/hp_coverage.svg")



def _run_one_path(path_i, freq_i, mode, model_type, max_trials, folder_name, do_BO):
    """Runs one path's full BO search (all trials/folds) + retrain + plots."""
    out_dir = f"{folder_name}/Bayesian_{model_type}_path{path_i}"
    if mode == "single":
        # set random seed for reproducibility
        tf.random.set_seed(42)
        results = [run_bayesian_optimization(path_i, freq_i, model_type, max_trials=max_trials, out_dir=out_dir, db_dir=folder_name, do_BO=do_BO)]
        
    elif mode == "duo":
        # set random seed for reproducibility
        tf.random.set_seed(42)
        results = [
            run_bayesian_optimization(path_i, freq_i, "fc_AE",  max_trials=max_trials, out_dir=out_dir, db_dir=folder_name, do_BO=do_BO),
            run_bayesian_optimization(path_i, freq_i, "CNN_AE", max_trials=max_trials, out_dir=out_dir, db_dir=folder_name, do_BO=do_BO),
        ]
    else:
        raise ValueError(f"Unknown MODE: {mode!r}. Use 'single' or 'duo'.")

    if do_BO:
        plot_all(results)

def _run_all_paths(FREQ_I, folder_name):
    for PATH_I in range(0, 28):
        if FREQ_I == 0 and PATH_I < 17:
            continue  
        if PATH_I == 0:  
            do_BO = True  # run BO for the first path to get best hyperparameters
        else:
            do_BO = False  # reuse best hyperparameters for other paths
        log_mem(f"before path {PATH_I} (parent)")
        _run_one_path(PATH_I, FREQ_I, MODE, MODEL_TYPE, MAX_TRIALS_AE, folder_name, do_BO)
        log_mem(f"after path {PATH_I} cleanup (parent)")  # should stay flat -- work happened in the child

def main():

    # Initialize new processes for each frequency to run the optimization

    for FREQ_I in range(0, 6):
        folder_name = str(TEST_RUN_DIR / f"Multi_path_BO_fixed_freq{FREQ_I}")
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(target=_run_all_paths, args=(FREQ_I, folder_name))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Subprocess for frequency {FREQ_I} failed with exit code {p.exitcode}")



if __name__ == "__main__":
    main()