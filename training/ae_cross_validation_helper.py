'''
Plotting and ensembling helpers for the autoencoder cross-validation loop in BO_AE and AE_train.py.

Includes:
- ClipLayer: a proper Layer wrapper for tf.clip_by_value, so the ensemble model saves and reloads without Keras's Lambda-deserialization restrictions.
- _drop_kwargs: monkey-patches Keras classes to drop unsupported kwargs (e.g. renorm, input_axes) that are present in the original code but not supported in the current Keras version.
- _predict_dataset: runs model.predict over every batch of a datastore, returns concatenated (sHI, reconstruction) arrays.

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
import tensorflow as tf
import matplotlib.pyplot as plt


class ClipLayer(tf.keras.layers.Layer):
    '''Elementwise tf.clip_by_value as a proper Layer (not a Lambda) so the ensemble
    model saves/reloads without Keras's Lambda-deserialization restrictions.'''

    def __init__(self, min_value=-10.0, max_value=10.0, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min_value, self.max_value)

    def get_config(self):
        config = super().get_config()
        config.update({"min_value": self.min_value, "max_value": self.max_value})
        return config


def _drop_kwargs(cls, *bad_keys):
    real_init = cls.__init__

    def patched_init(self, *args, **kwargs):
        for key in bad_keys:
            kwargs.pop(key, None)
        real_init(self, *args, **kwargs)

    cls.__init__ = patched_init

# This is in case some models were built with older Keras versions
_drop_kwargs(tf.keras.initializers.HeNormal, "input_axes", "output_axes")
_drop_kwargs(tf.keras.layers.BatchNormalization, "renorm", "renorm_clipping", "renorm_momentum")
_CompatHeNormal = tf.keras.initializers.HeNormal  

def _predict_dataset(model, ds):
    '''Runs model.predict over a datastore in one call, returns (sHI, reconstruction) arrays.'''
    shi_idx = model.output_names.index('sHI')
    rec_idx = model.output_names.index('reconstruction')
    pred = model.predict(ds, verbose=0)
    shi = np.asarray(pred[shi_idx])
    rec = np.asarray(pred[rec_idx])
    return shi.reshape(shi.shape[0], -1), rec


def plot_sHI_cv_fold(fold_model, ds_dict, States_dict, panels, held_out_panel, save_path):
    '''One figure, 4 subplots (one per base panel): sHI vs state predicted by a single
    cross-validation fold's model -- the fold that held `held_out_panel` out of training.'''
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    for i, panel in enumerate(panels):
        sHI, _ = _predict_dataset(fold_model, ds_dict[panel])
        states = States_dict[panel]
        axs[i].scatter(states, sHI[:, 0], alpha=0.7, s=15,
                        color='orange' if panel == held_out_panel else 'steelblue')
        axs[i].set_title(f'Panel {panel}' + (' (held out)' if panel == held_out_panel else ''))
        axs[i].set_xlabel('State idx')
        axs[i].set_ylabel('sHI')
        axs[i].grid(True)
    fig.suptitle(f'sHI vs State — fold validated on {held_out_panel}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_reconstruction_cv_fold(fold_model, ds_dict, States_dict, panels, held_out_panel, state_idx, save_path):
    '''One figure, 4 subplots (one per base panel): example input-vs-reconstructed signal
    at a given state, from a single cross-validation fold's model.'''
    rec_idx = fold_model.output_names.index('reconstruction')
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    for i, panel in enumerate(panels):
        states = States_dict[panel]
        offset = 0
        for x, _ in ds_dict[panel]:
            batch_size = x.shape[0]
            batch_states = states[offset:offset + batch_size]
            if np.any(batch_states == state_idx):
                local_idx = int(np.where(batch_states == state_idx)[0][0])
                pred = fold_model.predict(x, verbose=0)
                rec = np.asarray(pred[rec_idx])
                x_arr = np.asarray(x)
                axs[i].plot(x_arr[local_idx].reshape(-1), color='blue', alpha=0.6, label='Input')
                axs[i].plot(rec[local_idx].reshape(-1), color='green', alpha=0.8, label='Predicted')
                break
            offset += batch_size
        title = f'Panel {panel}' + (' (held out)' if panel == held_out_panel else '') + f', state={state_idx}'
        axs[i].set_title(title)
        axs[i].legend()
        axs[i].grid(True)
    fig.suptitle(f'Reconstruction — fold validated on {held_out_panel}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def _normalization_mean_variance_axis(norm_stats, input_shape):
    mean = norm_stats["feature_means"].astype(np.float32)
    variance = norm_stats["feature_stds"].astype(np.float32) ** 2

    if mean.shape == input_shape:
        axis = tuple(range(1, len(input_shape) + 1)) # all non-batch axes are Normalization axes (so every feature and channel pair gets its own stat)
        return mean, variance, axis

    # For cases where input is (n_feat,1) and mean is (n_feat,) -- reshape to (n_feat,1) so it broadcasts correctly across the channel axis.
    bshape = [mean.shape[0], 1]
    return mean.reshape(bshape), variance.reshape(bshape), axis


def build_ensemble_ae(fold_entries):
    '''Combines the per-fold cross-validation models into one averaging ensemble.

    Each fold's model was trained on features normalized with that fold's own mean/std
    (fit only on that fold's 3 training panels), so the ensemble can't just average raw
    model outputs -- it re-normalizes the shared raw input separately per branch, and
    de-normalizes each branch's reconstruction back to raw units before averaging.

    fold_entries: list of (panel_held_out, keras_model, norm_stats) tuples.
    '''
    input_shape = fold_entries[0][1].input_shape[1:]
    inp = tf.keras.Input(shape=input_shape, name="raw_input")

    shi_branches, rec_branches, latent_branches = [], [], []
    for panel, model, norm_stats in fold_entries:
        
        shi_idx = model.output_names.index("sHI")
        rec_idx = model.output_names.index("reconstruction")
        latent_output = model.get_layer("latent_space").output
        model = tf.keras.Model(inputs=model.input, outputs=[model.output[shi_idx], model.output[rec_idx], latent_output], name=model.name)
        shi_idx, rec_idx, latent_idx = 0, 1, 2

        """
        Every fold's model was built by the same factory function (build_CNN_AE_features/ build_fc_AE_features), so their internal layers have the same names. 
        This causes errors when building ensemble model, so the layer names must be prefixed with the fold's held-out panel name (to keep them unique)
        """
        prefix = f"fold_{panel}_"
        for layer in model.layers:
            layer.name = prefix + layer.name
        model.name = prefix + model.name

        # Prepare normalization and de-normalization layers 
        mean, variance, norm_axis = _normalization_mean_variance_axis(norm_stats, input_shape)
        norm_layer = tf.keras.layers.Normalization(axis=norm_axis, mean=mean, variance=variance, name=f"norm_{panel}")
        denorm_layer = tf.keras.layers.Normalization(axis=norm_axis, mean=mean, variance=variance, invert=True, name=f"denorm_{panel}")

        x_norm = norm_layer(inp)
        x_norm = ClipLayer(-10.0, 10.0, name=f"clip_{panel}")(x_norm)

        outs = model(x_norm)
        rec_denorm = denorm_layer(outs[rec_idx])

        shi_branches.append(outs[shi_idx])
        rec_branches.append(rec_denorm)
        latent_branches.append(outs[latent_idx])

    shi_avg = tf.keras.layers.Average(name="sHI")(shi_branches)
    rec_avg = tf.keras.layers.Average(name="reconstruction")(rec_branches)
    latent_avg = tf.keras.layers.Average(name="latent_space")(latent_branches)
    return tf.keras.Model(inputs=inp, outputs=[shi_avg, rec_avg, latent_avg], name="ensemble_ae")


def plot_ensemble_sHI(ensemble_model, ref_ds_dict, ref_norm_stats, States_dict, panels, save_path):
    '''sHI vs state predicted by the ensemble, one subplot per base panel.'''
    input_shape = ensemble_model.input_shape[1:]
    mean_b, variance_b, _ = _normalization_mean_variance_axis(ref_norm_stats, input_shape)
    std_b = np.sqrt(variance_b)

    shi_idx = ensemble_model.output_names.index("sHI")

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    for i, panel in enumerate(panels):
        states = States_dict[panel]
        all_sHI = []
        for x_norm, _ in ref_ds_dict[panel]:
            x_norm = np.asarray(x_norm).reshape((-1,) + input_shape)
            x_raw = x_norm * std_b + mean_b
            pred = ensemble_model.predict(x_raw, verbose=0)
            all_sHI.append(np.asarray(pred[shi_idx]).reshape(-1))
        sHI = np.concatenate(all_sHI)
        axs[i].scatter(states, sHI, alpha=0.7, s=15, color='purple')
        axs[i].set_title(f'Panel {panel}')
        axs[i].set_xlabel('State idx')
        axs[i].set_ylabel('sHI (ensemble)')
        axs[i].grid(True)
    fig.suptitle('Ensemble sHI vs State')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
