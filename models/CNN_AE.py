import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "tools", "training", "intermediate_results_check", "results_analysis"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import tensorflow as tf
import numpy as np

from config import DEFAULT_GCN_FEATURES, DEFAULT_K_SPARSE, DEFAULT_N_FEATURES


class KSparse(tf.keras.layers.Layer):
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def call(self, inputs):
        k = tf.minimum(self.k, tf.shape(inputs)[-1])
        values, _ = tf.math.top_k(tf.abs(inputs), k=k)
        kth = values[..., -1:]
        mask = tf.cast(tf.greater_equal(tf.abs(inputs), kth), inputs.dtype)
        return inputs * mask


class ExpandLastDim(tf.keras.layers.Layer):
    '''tf.expand_dims(x, -1) as a proper Layer -- a Lambda wrapping this doesn't
    reload reliably once nested inside another model (e.g. an averaging ensemble).'''

    def call(self, inputs):
        return tf.expand_dims(inputs, -1)


class SqueezeLastDim(tf.keras.layers.Layer):
    '''tf.squeeze(x, axis=-1) as a proper Layer -- see ExpandLastDim.'''

    def call(self, inputs):
        return tf.squeeze(inputs, axis=-1)


def build_fc_AE_features(params):

	input_size = params.get("input_size", 19+14)  # total input width (single or doubled for benchmark)
	n_features = params.get("n_features", input_size)  # single-channel feature count
	k_sparse = params.get("k_sparse", DEFAULT_K_SPARSE)
	latent_dim = params.get("latent_dim", 16)
	drop_rate = params.get("drop_rate", 0.0)
	has_benchmark = (input_size == n_features * 2)  # True when benchmark features are appended

	he = tf.keras.initializers.HeNormal()
	narrow = tf.keras.initializers.TruncatedNormal(stddev=0.05)
	reg = tf.keras.regularizers.l2(params.get("l2_reg", 1e-4))
	

	if has_benchmark:
		# subtract benchmark features from signal features before encoding
		inp = tf.keras.Input(shape=(n_features, 2), name="input")
		sig   = tf.keras.layers.Lambda(lambda t: t[:, :, 0], name="signal_feats")(inp)
		bench = tf.keras.layers.Lambda(lambda t: t[:, :, 1], name="bench_feats")(inp)
		x = tf.keras.layers.Subtract(name="diff_feats")([sig, bench])
	else:
		inp = tf.keras.Input(shape=(input_size,), name="input")
		x = inp

	x = tf.keras.layers.Dense(latent_dim, kernel_initializer=he, kernel_regularizer=reg, name="enc_dense")(x)
	x = tf.keras.layers.BatchNormalization(name="latent_bn")(x)
	x = tf.keras.layers.Activation("elu", name="enc_act")(x)
	x = tf.keras.layers.Dropout(drop_rate, name="latent_dropout")(x)
	k_sparse_layer = KSparse(k_sparse, name="latent_space")(x)

	lat = tf.keras.layers.Dense(1, activation=None, kernel_initializer=narrow, bias_initializer="zeros", kernel_regularizer=None, name="sHI")(k_sparse_layer)

	rec_size = n_features if has_benchmark else input_size
	rec = tf.keras.layers.Dense(latent_dim, kernel_initializer=he, kernel_regularizer=reg, name="dec_dense")(k_sparse_layer)
	rec = tf.keras.layers.BatchNormalization(name="dec_bn")(rec)
	rec = tf.keras.layers.Activation("elu", name="dec_act")(rec)
	rec = tf.keras.layers.Dense(rec_size, activation=None, kernel_initializer=he, bias_initializer="zeros", kernel_regularizer=None, name="reconstruction")(rec)

	model = tf.keras.Model(inputs=inp, outputs=[lat, rec], name="fc_AE_features")
	return model

def build_CNN_AE_features(params):
	
	n_feat     = params.get("n_features", DEFAULT_N_FEATURES)
	n_channels = params.get("n_channels", 1)   # 1 without benchmark, 2 with
	filters_bench    = params.get("filters_bench", 16)
	filters_path = params.get("filters_path", 16)
	latent_dim = params.get("latent_dim", DEFAULT_GCN_FEATURES)
	k_sparse   = params.get("k_sparse", DEFAULT_K_SPARSE)
	drop_rate  = params.get("drop_rate", 0.3)
	conv_out_h = n_feat//2  # encoder Conv2D output height (padding="valid")

	he     = tf.keras.initializers.HeNormal()
	narrow = tf.keras.initializers.TruncatedNormal(stddev=0.05)
	reg    = tf.keras.regularizers.l2(params.get("l2_reg", 1e-4))

	inp = tf.keras.Input(shape=(n_feat, n_channels), name="input")

	# ── Encoder ──────────────────────────────────────────────────────────
	# Add channel dim so Conv2D sees (n_feat, n_channels, 1)
	x = ExpandLastDim(name="expand")(inp)

	x = tf.keras.layers.Conv2D(filters_bench, kernel_size=[1, n_channels], padding="valid", strides=[1, n_channels],
	                           activation=None, kernel_initializer=he,
	                           kernel_regularizer=reg, name="bench_comp")(x)
	cx = tf.keras.layers.BatchNormalization(name="bench_bn")(x)
	cx = tf.keras.layers.Activation("elu", name="bench_act")(cx)


	x = tf.keras.layers.Conv2D(filters_path, kernel_size=[2, 1], padding="valid", strides=[2, 1],
	                           activation=None, kernel_initializer=he,
	                           kernel_regularizer=reg, name="dir_comp")(cx)
	cx = tf.keras.layers.BatchNormalization(name="dir_bn")(x)
	cx = tf.keras.layers.Activation("elu", name="dir_act")(cx)

	x = tf.keras.layers.Flatten(name="flatten")(cx)               # (batch, n_feat*filters)
	x = tf.keras.layers.Dense(latent_dim, kernel_initializer=he,
	                          kernel_regularizer=reg, name="enc_dense")(x)
	x = tf.keras.layers.BatchNormalization(name="lat_bn")(x)
	x = tf.keras.layers.Activation("elu", name="lat_act")(x)
	x = tf.keras.layers.Dropout(drop_rate, name="enc_dropout")(x)

	z = KSparse(k_sparse, name="latent_space")(x)                # (batch, latent_dim)

	# ── sHI head ─────────────────────────────────────────────────────────
	lat = tf.keras.layers.Dense(1, activation=None, kernel_initializer=narrow,
	                            kernel_regularizer=reg, name="sHI")(z)

	# ── Decoder (symmetric) ───────────────────────────────────────────────
	x = tf.keras.layers.Dense(conv_out_h * filters_path, kernel_initializer=he,
	                          kernel_regularizer=reg, name="dec_dense")(z)
	x = tf.keras.layers.BatchNormalization(name="dec_bn")(x)
	x = tf.keras.layers.Activation("elu", name="dec_act")(x)
	x = tf.keras.layers.Dropout(drop_rate, name="dec_dropout")(x)
	x = tf.keras.layers.Reshape((conv_out_h, 1, filters_path), name="reshape")(x)

	x = tf.keras.layers.Conv2DTranspose(filters_path, kernel_size=[1, 1], strides=[1, 1], padding="valid",
	                                    activation=None, kernel_initializer=he,
	                                    kernel_regularizer=reg, name="dec_conv")(x)
	x = tf.keras.layers.BatchNormalization(name="dec_bn2")(x)
	x = tf.keras.layers.Activation("elu", name="dec_act2")(x)

	x = tf.keras.layers.Conv2DTranspose(filters_bench, kernel_size=[2, 1], strides=[2, 1], padding="valid",
	                                    activation=None, kernel_initializer=he,
	                                    kernel_regularizer=reg, name="dec_conv1")(x)
	x = tf.keras.layers.BatchNormalization(name="dec_bn3")(x)
	x = tf.keras.layers.Activation("elu", name="dec_act3")(x)

	x = tf.keras.layers.Conv2DTranspose(1, kernel_size=[1, n_channels], padding="valid",
	                                    activation=None, kernel_initializer=he,
	                                    kernel_regularizer=reg, name="dec_conv2")(x)

	# shape: (batch, n_feat, n_channels, 1) → squeeze last dim
	rec = SqueezeLastDim(name="reconstruction")(x)

	model = tf.keras.Model(inputs=inp, outputs=[lat, rec], name="CNN_AE_features")
	return model


	

