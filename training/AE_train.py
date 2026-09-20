"""This file defines the process of training of the autoencoders with cross validation (among 103, 104, 105 and 109)"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "tools", "training", "intermediate_results_check", "results_analysis"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import random

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import pandas as pd

from CNN_AE import build_fc_AE_features, build_CNN_AE_features
from create_datastores import prepare_datastores
from ae_cross_validation_helper import (
    plot_sHI_cv_fold, plot_reconstruction_cv_fold, build_ensemble_ae, plot_ensemble_sHI,
)
from config import (
    BASE_PANELS, MODEL_DATABASE_XLSX_FEATURES, DEFAULT_FREQ_INDEX,
    DEFAULT_K_SPARSE, DEFAULT_N_FEATURES, MODEL_TRAIN_RESULTS_DIR,
    CROSS_VALIDATION_RESULTS_AE_DIR, TRAIN_PANELS, VAL_PANELS, TEST_PANELS, LR
)


def monotonicity_loss(y_true, y_pred):

    vae_seed = 42
    random.seed(vae_seed)
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    y_flat = tf.reshape(y_pred, [-1])
    batch_size = tf.shape(y_flat)[0]

    diff = y_flat[1:] - y_flat[:-1]  # consecutive differences, shape (batch-1,)
    diff = diff +tf.ones(batch_size-1, dtype=tf.float32) *10-  0*tf.random.normal([batch_size-1], mean=0.0, stddev=1.0)  # Shift to ensure positive values for monotonic increase
    diff = tf.pow(diff, 2)  # Square the differences to penalize negative values more heavily

    baseline = 10**2
    loss = tf.reduce_mean(diff) - baseline  # mean (not sum) so the loss scale is independent of batch_size
    return loss

def mse_excl_benchmark(y_true, y_pred):
    signal_true = y_true[:,:,0] # Assuming that the shape is batch_size x n_features x n_channels and that the benchmark features are in the second channel
    signal_pred = y_pred[:,:,0]
    mse = tf.reduce_mean(tf.square(signal_true - signal_pred))
    return mse

def bench_diff_mse(y_true, y_pred):
    bench_true = y_true[:,:,1] # Assuming that the shape is batch_size x n_features x n_channels and that the benchmark features are in the second channel
    signal_true = y_true[:,:,0]
    diff_true = signal_true - bench_true

    return tf.reduce_mean(tf.square(diff_true - (y_pred[:,:,0] - y_pred[:,:,1])))

def model_train_features(
        net_type='CNN_AE',         # 'fc_AE' or 'CNN_AE'
        include_benchmark=False,
        enable_diff=False,  # Only relevant for CNN_AE; if True, the model will be trained on the difference between signal and benchmark features
        params=None,
        learning_rate=LR,
        loss_weights=None,
        epochs=50,
        base_batch_size=30,
        test_batch_size=1,
        path_i=0,
        frequency_i=DEFAULT_FREQ_INDEX,
        train_ds_names=TRAIN_PANELS,
        val_ds_names=VAL_PANELS,
        test_ds_names=TEST_PANELS,
        results_dir=str(MODEL_TRAIN_RESULTS_DIR / "features"),
        seed=None,
        filepath=None
    ):
    '''
    Trains a features-based autoencoder (FC or CNN) on precomputed time-frequency features.
    net_type: 'fc_AE' uses build_fc_AE_features (flat input),
              'CNN_AE' uses build_CNN_AE_features (2-D feature × channel input).
    include_benchmark: if True, signal+benchmark features are concatenated (doubles n_channels for CNN).
    '''

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    N_FEAT     = DEFAULT_N_FEATURES
    n_channels = 2 if (include_benchmark and net_type == 'CNN_AE') else 1
    flat_size  = N_FEAT * n_channels          # 33 or 66

    # When diff_bench is active (fc_AE + benchmark), the datastore pre-computes the difference between signal and benchmark and returns a flat (N_FEAT,) vector, not (N_FEAT, 2).
    enable_diff = (net_type == 'fc_AE' and include_benchmark) or (net_type == 'CNN_AE' and enable_diff)
    model_input_size = N_FEAT if enable_diff else flat_size

    # Build model
    if net_type == 'fc_AE':
        default_params = {"k_sparse": DEFAULT_K_SPARSE, "input_size": model_input_size, 'n_features': N_FEAT}
        if params is None:
            params = default_params
        else:
            params["input_size"] = model_input_size   # must match datastore output shape
            params.setdefault("n_features", N_FEAT)
        model = build_fc_AE_features(params)

    elif net_type == 'CNN_AE':
        default_params = {"k_sparse": DEFAULT_K_SPARSE, "n_features": N_FEAT, "n_channels": n_channels,
                          "filters_bench": 16, "filters_path": 16, "latent_dim": 16, "drop_rate":0.2}
        if params is None:
            params = default_params
        else:
            params.setdefault("n_features", N_FEAT)
            params.setdefault("n_channels", n_channels)
        model = build_CNN_AE_features(params)

    else:
        raise ValueError(f"Unknown net_type '{net_type}'. Choose 'fc_AE' or 'CNN_AE'.")

    Vlearning_rate = learning_rate
    Voptimizer = tf.keras.optimizers.Adam(learning_rate=Vlearning_rate)
    Vepochs = epochs

    Vloss_fun = {'reconstruction': 'mse', 'sHI': monotonicity_loss}
    Vloss_weights = loss_weights if loss_weights is not None else {'reconstruction': 1.0, 'sHI': 2.0}

    train_dataset, val_dataset, test_dataset, ds_dict, RUL_dict, States_dict, norm_stats = prepare_datastores(
        path_i, frequency_i, base_batch_size, test_batch_size,
        train_ds_names, val_ds_names, test_ds_names,
        include_benchmark=include_benchmark, diff_bench=enable_diff
    )


    model.compile(
        optimizer=Voptimizer,
        loss=Vloss_fun,
        run_eagerly=False,
        loss_weights=Vloss_weights,
    )

    print("Starting training...")

    # Define callbacks for early stopping and learning rate reduction
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=105, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5,
        ),
    ]

    # Train the model and capture the training history
    history = model.fit(train_dataset, epochs=Vepochs, verbose=0, validation_data=val_dataset, callbacks=callbacks)

   
    final_loss_idx = np.argmin(history.history['val_loss'])
    final_loss = history.history['loss'][final_loss_idx]
    rec_loss_at_final = history.history['reconstruction_loss'][final_loss_idx]
    lat_loss_at_final = history.history['sHI_loss'][final_loss_idx]

    val_final_loss = history.history['val_loss'][final_loss_idx]
    val_rec_loss_at_final = history.history['val_reconstruction_loss'][final_loss_idx]
    val_lat_loss_at_final = history.history['val_sHI_loss'][final_loss_idx]

    # Save the trained model
    data_time = pd.Timestamp.now().strftime("%d-%m-%H-%M")
    bench_tag = "_benchmark" if include_benchmark else ""
    if filepath is not None:
        # Connect filepath to results_dir if it's not an absolute path
        if not os.path.isabs(filepath):
            filepath = os.path.join(results_dir, filepath)
        path = filepath
    else:
        path = f"{results_dir}-{data_time}-{net_type}_features{bench_tag}_{final_loss:.4f}.keras"

    model.save(path)
    print(f"Model saved as {path}\n")

    # 1. Capture model summary as a string for the text file
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))

    # 2. Get the number of trainable parameters
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

    # 3. Create the info dictionary
    model_info = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "net_type": f"{net_type}_features{'_benchmark' if include_benchmark else ''}",
        "final_loss": float(f"{final_loss:.6f}"),
        "val_rec_loss": float(f"{history.history[f'val_reconstruction_loss'][-1]:.6f}"),
        "val_lat_loss": float(f"{history.history['val_sHI_loss'][-1]:.6f}"),
        "epochs": Vepochs,
        "batch_size": base_batch_size,
        "learning_rate": Vlearning_rate,
        "k_sparse": params.get("k_sparse", "N/A"),
        "trainable_params": trainable_count,
        "filters_bench": params.get("filters_bench", "N/A"),
        "filters_path": params.get("filters_path", "N/A"),
        "latent_dim": params.get("latent_dim", "N/A"),
        "drop_rate": params.get("drop_rate", "N/A"),
        "kernel_size": params.get("kernel_size", "N/A"),
        "optimizer": str(Voptimizer.__class__.__name__),
        "loss_weights": str(Vloss_weights),
        "path_index": path_i,
        "frequency_index": frequency_i,
        "train_ds_names": train_ds_names,
        "seed": seed,
        "model_file": path
    }

    # --- SAVE TO EXCEL DATABASE ---
    excel_path = str(MODEL_DATABASE_XLSX_FEATURES)

    # Create a DataFrame from the single run
    df_new = pd.DataFrame([model_info])

    if not os.path.exists(excel_path):
        # If file doesn't exist, create it
        df_new.to_excel(excel_path, index=False)
        print(f"Created new database: {excel_path}\n")
    else:
        # If exists, append the new row
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            # Load existing data to find the next empty row
            try:
                existing_df = pd.read_excel(excel_path)
                df_new.to_excel(writer, index=False, header=False, startrow=len(existing_df) + 1)
            except Exception as e:
                # Fallback if file is corrupted/empty
                df_new.to_excel(excel_path, index=False)
        print(f"Updated database at {excel_path}\n")

    return model, history, final_loss, rec_loss_at_final, lat_loss_at_final, val_final_loss, val_rec_loss_at_final, val_lat_loss_at_final, model_info, ds_dict, RUL_dict, States_dict, train_ds_names, val_ds_names, test_ds_names, results_dir, norm_stats


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    net_type = 'CNN_AE'  # Options: 'fc_AE', 'CNN_AE'
    basic_panels = BASE_PANELS  # panels used for cross-validation (each held out once)
    benchmark = True
    enable_diff = False # Only relevant for CNN_AE; if True, the model will be trained on the difference between signal and benchmark features
    state_idx_for_recon = 0  # example state shown in the per-fold reconstruction plots
    path_indexes = [0]  # path indexes to iterate over; adjust as needed

    losses_of_paths = {panel: np.zeros(len(path_indexes)) for panel in basic_panels}
    rec_losses_of_paths = {panel: np.zeros(len(path_indexes)) for panel in basic_panels}
    lat_losses_of_paths = {panel: np.zeros(len(path_indexes)) for panel in basic_panels}
    val_losses_of_paths = {panel: np.zeros(len(path_indexes)) for panel in basic_panels}
    val_rec_losses_of_paths = {panel: np.zeros(len(path_indexes)) for panel in basic_panels}
    val_lat_losses_of_paths = {panel: np.zeros(len(path_indexes)) for panel in basic_panels}

    average_final_loss = np.zeros(len(path_indexes))
    val_average_final_loss = np.zeros(len(path_indexes))

    for path_i, p_idx in enumerate(path_indexes):
        path_dir = os.path.join(str(CROSS_VALIDATION_RESULTS_AE_DIR), f"path_{p_idx}")
        os.makedirs(path_dir, exist_ok=True)

        final_loss_list = np.zeros(len(basic_panels))
        val_final_loss_list = np.zeros(len(basic_panels))
        rec_train_loss_list, lat_train_loss_list = [], []
        rec_val_loss_list, lat_val_loss_list = [], []
        fold_entries = []  # (panel_held_out, model, norm_stats) -- used to build the ensemble
        last_ds_dict, last_States_dict = None, None

        t_start = pd.Timestamp.now()
        for i, panel in enumerate(basic_panels):
            print(f"Validating using panel {panel}...")
            train_ds_names = [p for p in basic_panels if p != panel]
            val_ds_names = [panel]

            
            params = {
                "k_sparse": DEFAULT_K_SPARSE,
                "k_sparse": DEFAULT_K_SPARSE, "n_features": DEFAULT_N_FEATURES, "n_channels": 2 if (benchmark ) else 1,
                        "filters_bench": 10, "filters_path": 10, "latent_dim": 16, "drop_rate":0.2
                # input_size is overridden by model_train_features to match the datastore shape
            }
            (model, history, final_loss, rec_loss_at_final, lat_loss_at_final,
                val_final_loss, val_rec_loss_at_final, val_lat_loss_at_final,
                model_info, ds_dict,  RUL_dict, States_dict,
                train_ds_names, val_ds_names, test_ds_names, results_dir, norm_stats) = model_train_features(
                net_type=net_type,         # switch to 'CNN_AE' to use the CNN variant
                include_benchmark=benchmark,
                enable_diff=enable_diff,
                epochs=200, learning_rate=LR,
                params=params, path_i=p_idx, train_ds_names=train_ds_names, val_ds_names=val_ds_names,
                seed=seed, results_dir=path_dir, filepath=f"Model_val_{panel}.keras",
            )
            recon_name, latent_name = 'reconstruction', 'sHI'

            plot_sHI_cv_fold(model, ds_dict, States_dict, basic_panels, panel,
                                save_path=os.path.join(path_dir, f"sHI_fold_val_{panel}.png"))
            plot_reconstruction_cv_fold(model, ds_dict, States_dict, basic_panels, panel, state_idx_for_recon,
                                            save_path=os.path.join(path_dir, f"reconstruction_fold_val_{panel}.png"))

            fold_entries.append((panel, model, norm_stats))
            last_ds_dict, last_States_dict = ds_dict, States_dict
            

            rec_train_loss_list.append(history.history[f'{recon_name}_loss'])
            lat_train_loss_list.append(history.history[f'{latent_name}_loss'])
            rec_val_loss_list.append(history.history[f'val_{recon_name}_loss'])
            lat_val_loss_list.append(history.history[f'val_{latent_name}_loss'])

            final_loss_list[i] = final_loss
            val_final_loss_list[i] = val_final_loss
            losses_of_paths[panel][path_i] = final_loss
            rec_losses_of_paths[panel][path_i] = rec_loss_at_final
            lat_losses_of_paths[panel][path_i] = lat_loss_at_final
            val_losses_of_paths[panel][path_i] = val_final_loss
            val_rec_losses_of_paths[panel][path_i] = val_rec_loss_at_final
            val_lat_losses_of_paths[panel][path_i] = val_lat_loss_at_final

        t_end = pd.Timestamp.now()
        print(f"Training and validation for path index {p_idx} completed in {t_end - t_start}.\n")
        print("Final losses for each panel:", final_loss_list)

        average_final_loss[path_i] = np.mean(final_loss_list)
        val_average_final_loss[path_i] = np.mean(val_final_loss_list)
        print(f"Average final loss across panels: {average_final_loss[path_i]:.4f}\n")

        # Learning curves: reconstruction loss (top row) + latent loss (bottom row), one column per panel
        plt.figure(figsize=(16, 6))
        for i, panel in enumerate(basic_panels):
            plt.subplot(2, 4, i + 1)
            plt.plot(rec_train_loss_list[i], label='Train')
            plt.plot(rec_val_loss_list[i], label='Val')
            plt.title(f'Recon Loss - Panel {panel}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

        for i, panel in enumerate(basic_panels):
            plt.subplot(2, 4, i + 5)
            plt.plot(lat_train_loss_list[i], label='Train')
            plt.plot(lat_val_loss_list[i], label='Val')
            plt.title(f'Latent Loss - Panel {panel}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(path_dir, f"learning_curves_{net_type}_{average_final_loss[path_i]:.4f}.png"))
        plt.close()

        # Ensemble of the 4 cross-validation folds
        if len(fold_entries) == len(basic_panels):
            ensemble_model = build_ensemble_ae(fold_entries)
            ensemble_model.save(os.path.join(path_dir, "ensemble_model.keras"))
            _, _, ref_norm_stats = fold_entries[0]
            plot_ensemble_sHI(
                ensemble_model, last_ds_dict, ref_norm_stats, last_States_dict, basic_panels,
                save_path=os.path.join(path_dir, "ensemble_sHI.png"),
            )


    # save the losses of all paths in a csv file for later analysis
    summary_dir = str(CROSS_VALIDATION_RESULTS_AE_DIR)
    os.makedirs(summary_dir, exist_ok=True)
    losses_df = pd.DataFrame({
        'path_index': path_indexes,
        'average_final_loss': average_final_loss,
        'val_average_final_loss': val_average_final_loss,
    })
    losses_df.to_csv(os.path.join(summary_dir, "losses_of_paths.csv"), index=False)

    # save losses per panel and path in separate csv files
    for panel in basic_panels:
        panel_df = pd.DataFrame({
            'path_index': path_indexes,
            'training_loss': losses_of_paths[panel],
            'validation_loss': val_losses_of_paths[panel],
            'training_recon_loss': rec_losses_of_paths[panel],
            'validation_recon_loss': val_rec_losses_of_paths[panel],
            'training_latent_loss': lat_losses_of_paths[panel],
            'validation_latent_loss': val_lat_losses_of_paths[panel],
        })
        panel_df.to_csv(os.path.join(summary_dir, f"losses_panel_{panel}.csv"), index=False)

    if len(path_indexes) > 1:
        # Plot final losses across panels for each path index
        plt.figure(figsize=(10, 6))
        plt.plot(path_indexes, average_final_loss, marker='o', label='Training')
        plt.plot(path_indexes, val_average_final_loss, marker='s', label='Validation')
        plt.title('Average Final Loss Across Panels vs Path Index')
        plt.xlabel('Path Index')
        plt.ylabel('Final Loss')
        plt.legend()
        plt.savefig(os.path.join(summary_dir, "final_loss_across_paths.png"))
        plt.show()

        # Plot 4 subplots for every validation panel, on each plot there is training and validation
        # performance for every path index. 3 such figures are created: total, reconstruction, latent loss.
        fig_total, axs_total = plt.subplots(2, 2, figsize=(12, 10))
        fig_rec, axs_rec = plt.subplots(2, 2, figsize=(12, 10))
        fig_lat, axs_lat = plt.subplots(2, 2, figsize=(12, 10))
        for i, panel in enumerate(basic_panels):
            row, col = divmod(i, 2)
            axs_total[row, col].scatter(path_indexes, losses_of_paths[panel], marker='o', label='Training Loss')
            axs_total[row, col].scatter(path_indexes, val_losses_of_paths[panel], marker='s', label='Validation Loss')
            axs_total[row, col].set_title(f'Panel {panel} - Total Loss')
            axs_total[row, col].set_xlabel('Path Index')
            axs_total[row, col].set_ylabel('Loss')
            axs_total[row, col].legend()

            axs_rec[row, col].scatter(path_indexes, rec_losses_of_paths[panel], marker='o', label='Training Recon Loss')
            axs_rec[row, col].scatter(path_indexes, val_rec_losses_of_paths[panel], marker='s', label='Validation Recon Loss')
            axs_rec[row, col].set_title(f'Panel {panel} - Reconstruction Loss')
            axs_rec[row, col].set_xlabel('Path Index')
            axs_rec[row, col].set_ylabel('Loss')
            axs_rec[row, col].legend()

            axs_lat[row, col].scatter(path_indexes, lat_losses_of_paths[panel], marker='o', label='Training Latent Loss')
            axs_lat[row, col].scatter(path_indexes, val_lat_losses_of_paths[panel], marker='s', label='Validation Latent Loss')
            axs_lat[row, col].set_title(f'Panel {panel} - Latent Loss')
            axs_lat[row, col].set_xlabel('Path Index')
            axs_lat[row, col].set_ylabel('Loss')
            axs_lat[row, col].legend()

        fig_total.savefig(os.path.join(summary_dir, "total_loss_across_paths.png"))
        fig_rec.savefig(os.path.join(summary_dir, "reconstruction_loss_across_paths.png"))
        fig_lat.savefig(os.path.join(summary_dir, "latent_loss_across_paths.png"))
        plt.show()
