'''
This code implements extraction of 19 time domain features and 14 frequency domain
features, each computed separately on the first and second half of the 4000-sample
time signal (split evenly at the midpoint) -- 66 features total per state.
The class inherits from the states class
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
from states import states
import tensorflow as tf
import tensorflow_probability as tfp
from config import BASE_PANELS, GRAPH_DATA_DIR, PANEL_123_SUBPANELS, mat_file_path, DEFAULT_FREQ_INDEX

class FeaturesExtractor(states):
    def __init__(self, mat_file):
        super().__init__(mat_file=mat_file)
        self.mat_file = mat_file


    # Shared helpers
    def _get_amp(self, s_idx, freq, p_idx, benchmark):
        return self.benchmark_amplitude(s_idx, freq, p_idx) if benchmark else self.amplitude(s_idx, freq, p_idx)

    def _split_halves(self, amp):
        '''Splits a time-domain signal into two equal halves along the time axis, to extract features from each direction along the path'''
        n = amp.shape[-1]
        mid = n // 2
        return amp[..., :mid], amp[..., mid:]

    # -------------------------------------------------------------- #
    # Time domain features T1–T19 (Table A.13)                            #
    # -------------------------------------------------------------- #
    def _skewness(self, amp):
        mean = tf.reduce_mean(amp, axis=-1, keepdims=True)
        std = tf.math.reduce_std(amp, axis=-1, keepdims=True)
        n = tf.cast(tf.shape(amp)[-1], amp.dtype)
        numerator = tf.reduce_sum(tf.pow(amp - mean, 3), axis=-1)
        denominator = tf.squeeze(tf.pow(std, 3), axis=-1) * (n - 1)
        return numerator / denominator

    def _kurtosis(self, amp):
        mean = tf.reduce_mean(amp, axis=-1, keepdims=True)
        std  = tf.math.reduce_std(amp, axis=-1, keepdims=True)
        n    = tf.cast(tf.shape(amp)[-1], amp.dtype)
        numerator   = tf.reduce_sum(tf.pow(amp - mean, 4), axis=-1)
        denominator = tf.squeeze(tf.pow(std, 4), axis=-1) * (n - 1)
        return numerator / denominator

    def _k_central_moment(self, amp, k):
        mean = tf.reduce_mean(amp, axis=-1, keepdims=True)
        return tf.reduce_mean(tf.pow(amp - mean, k), axis=-1)

    def extract_mean(self, s_idx, freq, p_idx, benchmark=False):
        half1, half2 = self._split_halves(self._get_amp(s_idx, freq, p_idx, benchmark))
        return tf.reduce_mean(half1, axis=-1), tf.reduce_mean(half2, axis=-1)

    def extract_std(self, s_idx, freq, p_idx, benchmark=False):
        half1, half2 = self._split_halves(self._get_amp(s_idx, freq, p_idx, benchmark))
        return tf.math.reduce_std(half1, axis=-1), tf.math.reduce_std(half2, axis=-1)

    def extract_root_amplitude(self, s_idx, freq, p_idx, benchmark=False):
        half1, half2 = self._split_halves(self._get_amp(s_idx, freq, p_idx, benchmark))
        feat1 = tf.pow(tf.reduce_mean(tf.sqrt(tf.abs(half1)), axis=-1), 2)
        feat2 = tf.pow(tf.reduce_mean(tf.sqrt(tf.abs(half2)), axis=-1), 2)
        return feat1, feat2

    def extract_rms(self, s_idx, freq, p_idx, benchmark=False):
        half1, half2 = self._split_halves(self._get_amp(s_idx, freq, p_idx, benchmark))
        return tf.sqrt(tf.reduce_mean(tf.square(half1), axis=-1)), tf.sqrt(tf.reduce_mean(tf.square(half2), axis=-1))

    def extract_rss(self, s_idx, freq, p_idx, benchmark=False):
        half1, half2 = self._split_halves(self._get_amp(s_idx, freq, p_idx, benchmark))
        return tf.sqrt(tf.reduce_sum(tf.pow(half1, 2), axis=-1)), tf.sqrt(tf.reduce_sum(tf.pow(half2, 2), axis=-1))

    def extract_peak(self, s_idx, freq, p_idx, benchmark=False):
        half1, half2 = self._split_halves(self._get_amp(s_idx, freq, p_idx, benchmark))
        return tf.reduce_max(tf.abs(half1), axis=-1), tf.reduce_max(tf.abs(half2), axis=-1)

    def extract_skweness(self, s_idx, freq, p_idx, benchmark=False):
        half1, half2 = self._split_halves(self._get_amp(s_idx, freq, p_idx, benchmark))
        return self._skewness(half1), self._skewness(half2)

    def extract_kurtosis(self, s_idx, freq, p_idx, benchmark=False):
        half1, half2 = self._split_halves(self._get_amp(s_idx, freq, p_idx, benchmark))
        return self._kurtosis(half1), self._kurtosis(half2)

    def extract_crest_factor(self, s_idx, freq, p_idx, benchmark=False):
        peak1, peak2 = self.extract_peak(s_idx, freq, p_idx, benchmark=benchmark)
        rms1, rms2 = self.extract_rms(s_idx, freq, p_idx, benchmark=benchmark)
        return peak1 / rms1, peak2 / rms2

    def extract_clearance_factor(self, s_idx, freq, p_idx, benchmark=False):
        peak1, peak2 = self.extract_peak(s_idx, freq, p_idx, benchmark=benchmark)
        ra1, ra2 = self.extract_root_amplitude(s_idx, freq, p_idx, benchmark=benchmark)
        return peak1 / ra1, peak2 / ra2

    def extract_shape_factor(self, s_idx, freq, p_idx, benchmark=False):
        half1, half2 = self._split_halves(self._get_amp(s_idx, freq, p_idx, benchmark))
        rms1, rms2 = self.extract_rms(s_idx, freq, p_idx, benchmark=benchmark)
        mean1 = tf.reduce_mean(tf.abs(half1), axis=-1)
        mean2 = tf.reduce_mean(tf.abs(half2), axis=-1)
        return rms1 / mean1, rms2 / mean2

    def extract_impulse_factor(self, s_idx, freq, p_idx, benchmark=False):
        half1, half2 = self._split_halves(self._get_amp(s_idx, freq, p_idx, benchmark))
        peak1, peak2 = self.extract_peak(s_idx, freq, p_idx, benchmark=benchmark)
        mean1 = tf.reduce_mean(tf.abs(half1), axis=-1)
        mean2 = tf.reduce_mean(tf.abs(half2), axis=-1)
        return peak1 / mean1, peak2 / mean2

    def extract_maxmin_difference(self, s_idx, freq, p_idx, benchmark=False):
        half1, half2 = self._split_halves(self._get_amp(s_idx, freq, p_idx, benchmark))
        feat1 = tf.reduce_max(half1, axis=-1) - tf.reduce_min(half1, axis=-1)
        feat2 = tf.reduce_max(half2, axis=-1) - tf.reduce_min(half2, axis=-1)
        return feat1, feat2

    def extract_k_central_moment(self, s_idx, freq, p_idx, k, benchmark=False):
        half1, half2 = self._split_halves(self._get_amp(s_idx, freq, p_idx, benchmark))
        return self._k_central_moment(half1, k), self._k_central_moment(half2, k)

    def extract_FM4(self, s_idx, freq, p_idx, benchmark=False):
        k4_1, k4_2 = self.extract_k_central_moment(s_idx, freq, p_idx, 4, benchmark=benchmark)
        std1, std2 = self.extract_std(s_idx, freq, p_idx, benchmark=benchmark)
        return k4_1 / tf.pow(std1, 4), k4_2 / tf.pow(std2, 4)

    def extract_median(self, s_idx, freq, p_idx, benchmark=False):
        half1, half2 = self._split_halves(self._get_amp(s_idx, freq, p_idx, benchmark))
        return tfp.stats.percentile(half1, 50.0, axis=-1), tfp.stats.percentile(half2, 50.0, axis=-1)


    # ------------------------------------------------------------------ #
    # Frequency domain features S1–S14  (Table A.14)                      #
    # f_k : frequency axis (Hz),  s_k : one-sided power spectrum |FFT|²   #
    # ------------------------------------------------------------------ #

    def _power_spectrum(self, s_idx, freq_idx, p_idx, sample_rate=1.0, benchmark=False, half=None):
        # s_idx may be ":" (all states → 2D) or an int (single state → 1D)
        x = self._get_amp(s_idx, freq_idx, p_idx, benchmark)
        if hasattr(x, 'numpy'):
            x = x.numpy()
        x = np.asarray(x, dtype=float)
        if half is not None:
            half1, half2 = self._split_halves(x)
            x = half1 if half == 1 else half2
        N = x.shape[-1]                              # signal length, not n_states
        s_k = np.abs(np.fft.rfft(x, axis=-1)) ** 2  # (..., N_f)
        f_k = np.fft.rfftfreq(N, d=1.0 / sample_rate)  # (N_f,)
        return f_k, s_k

    def _S1(self, f, s):
        """S1 = mean(S)"""
        return np.mean(s, axis=-1)

    def _S2(self, f, s):
        """S2 = var(S)"""
        return np.var(s, axis=-1)

    def _S3(self, f, s):
        """S3 = skewness(S)"""
        from scipy.stats import skew
        return skew(s, axis=-1)

    def _S4(self, f, s):
        """S4 = kurtosis(S)"""
        from scipy.stats import kurtosis
        return kurtosis(s, axis=-1)

    def _S5(self, f, s):
        """S5 = X_fc = Σ(f_k · s_k) / Σ(s_k)  (centroid frequency)"""
        return np.sum(f * s, axis=-1) / np.sum(s, axis=-1)

    def _S6(self, f, s):
        """S6 = sqrt(Σ((f_k − S5)² · s_k) / N_f)"""
        s5 = np.sum(f * s, axis=-1, keepdims=True) / np.sum(s, axis=-1, keepdims=True)
        return np.sqrt(np.sum((f - s5) ** 2 * s, axis=-1) / len(f))

    def _S7(self, f, s):
        """S7 = X_rmsf = sqrt(Σ(f_k² · s_k) / Σ(s_k))"""
        return np.sqrt(np.sum(f ** 2 * s, axis=-1) / np.sum(s, axis=-1))

    def _S8(self, f, s):
        """S8 = sqrt(Σ(f_k⁴ · s_k) / Σ(f_k² · s_k))"""
        return np.sqrt(np.sum(f ** 4 * s, axis=-1) / np.sum(f ** 2 * s, axis=-1))

    def _S9(self, f, s):
        """S9 = Σ(f_k² · s_k) / sqrt(Σ(s_k) · Σ(f_k⁴ · s_k))"""
        return np.sum(f ** 2 * s, axis=-1) / np.sqrt(np.sum(s, axis=-1) * np.sum(f ** 4 * s, axis=-1))

    def _S10(self, f, s):
        """S10 = S6 / S5  (normalised RMS spectral deviation)"""
        s5 = np.sum(f * s, axis=-1, keepdims=True) / np.sum(s, axis=-1, keepdims=True)
        s6 = np.sqrt(np.sum((f - s5) ** 2 * s, axis=-1) / len(f))
        return s6 / s5.squeeze(-1)

    def _S11(self, f, s):
        """S11 = Σ((f_k − S5)³ · s_k) / (N_f · S6³)"""
        n = len(f)
        s5 = np.sum(f * s, axis=-1, keepdims=True) / np.sum(s, axis=-1, keepdims=True)
        s6 = np.sqrt(np.sum((f - s5) ** 2 * s, axis=-1) / n)
        return np.sum((f - s5) ** 3 * s, axis=-1) / (n * s6 ** 3)

    def _S12(self, f, s):
        """S12 = Σ((f_k − S5)⁴ · s_k) / (N_f · S6⁴)"""
        n = len(f)
        s5 = np.sum(f * s, axis=-1, keepdims=True) / np.sum(s, axis=-1, keepdims=True)
        s6 = np.sqrt(np.sum((f - s5) ** 2 * s, axis=-1) / n)
        return np.sum((f - s5) ** 4 * s, axis=-1) / (n * s6 ** 4)

    def _S13(self, f, s):
        """S13 = Σ(sqrt(|f_k − S5|) · s_k) / (N_f · sqrt(S6))"""
        n = len(f)
        s5 = np.sum(f * s, axis=-1, keepdims=True) / np.sum(s, axis=-1, keepdims=True)
        s6 = np.sqrt(np.sum((f - s5) ** 2 * s, axis=-1) / n)
        return np.sum(np.sqrt(np.abs(f - s5)) * s, axis=-1) / (n * np.sqrt(s6))

    def _S14(self, f, s):
        """S14 = sqrt(Σ((f_k − S5)² · s_k) / Σ(s_k))"""
        s5 = np.sum(f * s, axis=-1, keepdims=True) / np.sum(s, axis=-1, keepdims=True)
        return np.sqrt(np.sum((f - s5) ** 2 * s, axis=-1) / np.sum(s, axis=-1))

    def _extract_S(self, formula, s_idx, freq, p_idx, sample_rate, benchmark):
        f1, s1 = self._power_spectrum(s_idx, freq, p_idx, sample_rate, benchmark=benchmark, half=1)
        f2, s2 = self._power_spectrum(s_idx, freq, p_idx, sample_rate, benchmark=benchmark, half=2)
        return formula(f1, s1), formula(f2, s2)

    def extract_S1(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        return self._extract_S(self._S1, s_idx, freq, p_idx, sample_rate, benchmark)

    def extract_S2(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        return self._extract_S(self._S2, s_idx, freq, p_idx, sample_rate, benchmark)

    def extract_S3(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        return self._extract_S(self._S3, s_idx, freq, p_idx, sample_rate, benchmark)

    def extract_S4(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        return self._extract_S(self._S4, s_idx, freq, p_idx, sample_rate, benchmark)

    def extract_S5(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        return self._extract_S(self._S5, s_idx, freq, p_idx, sample_rate, benchmark)

    def extract_S6(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        return self._extract_S(self._S6, s_idx, freq, p_idx, sample_rate, benchmark)

    def extract_S7(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        return self._extract_S(self._S7, s_idx, freq, p_idx, sample_rate, benchmark)

    def extract_S8(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        return self._extract_S(self._S8, s_idx, freq, p_idx, sample_rate, benchmark)

    def extract_S9(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        return self._extract_S(self._S9, s_idx, freq, p_idx, sample_rate, benchmark)

    def extract_S10(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        return self._extract_S(self._S10, s_idx, freq, p_idx, sample_rate, benchmark)

    def extract_S11(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        return self._extract_S(self._S11, s_idx, freq, p_idx, sample_rate, benchmark)

    def extract_S12(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        return self._extract_S(self._S12, s_idx, freq, p_idx, sample_rate, benchmark)

    def extract_S13(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        return self._extract_S(self._S13, s_idx, freq, p_idx, sample_rate, benchmark)

    def extract_S14(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        return self._extract_S(self._S14, s_idx, freq, p_idx, sample_rate, benchmark)

    def extract_all_features(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False, diff = False, cache_dir=None):

        if benchmark and diff:
            raise ValueError("Cannot set both benchmark and diff to True. Please choose one.")

        if p_idx == ":":
            print(f"Extracting features for all paths at freq {freq} Hz...")
            if cache_dir is not None:
                panel_name = Path(self.mat_file).stem
                if benchmark:
                    cache_file = Path(cache_dir) / f"{panel_name}_freq{freq}_all_paths_benchmark_full.npy"
                elif diff:
                    cache_file = Path(cache_dir) / f"{panel_name}_freq{freq}_all_paths_diff_full.npy"
                else:
                    cache_file = Path(cache_dir) / f"{panel_name}_freq{freq}_all_paths_full.npy"
                if cache_file.exists():
                    return np.load(str(cache_file))
                else:
                    # extract features for all paths and save to cache
                    all_features = []

                    for p in range(0,28):
                        if benchmark or diff:
                            features_p = self.extract_all_features(s_idx, freq, p_idx=p, sample_rate=sample_rate, benchmark=False, cache_dir=None)
                            features_p_ben = self.extract_all_features(s_idx, freq, p_idx=p, sample_rate=sample_rate, benchmark=True, cache_dir=None)
                            if benchmark:
                                features_p = np.concatenate([features_p, features_p_ben], axis=-1)
                            elif diff:
                                features_p = features_p - features_p_ben
                        else:
                            features_p = self.extract_all_features(s_idx, freq, p_idx=p, sample_rate=sample_rate, benchmark=False, cache_dir=None)
                        all_features.append(features_p)
                    all_features = np.stack(all_features, axis=0)  # (n_paths, n_states, n_features)

                    # create cache directory if it doesn't exist
                    Path(cache_dir).mkdir(parents=True, exist_ok=True)
                    np.save(str(cache_file), all_features)
                    return all_features
            else:
                input(f"Extracting features without caching for all paths at freq {freq} Hz... Press Enter to continue.")
                all_features = []

                for p in range(0,28):
                    features_p = self.extract_all_features(s_idx, freq, p_idx=p, sample_rate=sample_rate, benchmark=benchmark, cache_dir=None)
                    all_features.append(features_p)
                all_features = np.stack(all_features, axis=0)  # (n_paths, n_states, n_features)
                return all_features         

        else:

            if cache_dir is not None:
                
                panel_name = Path(self.mat_file).stem
                if benchmark:
                    cache_file = Path(cache_dir) / f"{panel_name}_freq{freq}_path{p_idx}_benchmark_full.npy"

                elif diff:
                    cache_file = Path(cache_dir) / f"{panel_name}_freq{freq}_path{p_idx}_diff_full.npy"
                else:
                    cache_file = Path(cache_dir) / f"{panel_name}_freq{freq}_path{p_idx}_full.npy"
                if cache_file.exists():
                    return np.load(str(cache_file))

            self.features = {}

            self.features['mean1'], self.features['mean2'] = self.extract_mean(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['std1'], self.features['std2'] = self.extract_std(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['root_amplitude1'], self.features['root_amplitude2'] = self.extract_root_amplitude(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['rms1'], self.features['rms2'] = self.extract_rms(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['rss1'], self.features['rss2'] = self.extract_rss(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['peak1'], self.features['peak2'] = self.extract_peak(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['skweness1'], self.features['skweness2'] = self.extract_skweness(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['kurtosis1'], self.features['kurtosis2'] = self.extract_kurtosis(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['crest_factor1'], self.features['crest_factor2'] = self.extract_crest_factor(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['clearance_factor1'], self.features['clearance_factor2'] = self.extract_clearance_factor(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['shape_factor1'], self.features['shape_factor2'] = self.extract_shape_factor(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['impulse_factor1'], self.features['impulse_factor2'] = self.extract_impulse_factor(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['maxmin_difference1'], self.features['maxmin_difference2'] = self.extract_maxmin_difference(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['3rd_central_moment1'], self.features['3rd_central_moment2'] = self.extract_k_central_moment(s_idx, freq, p_idx, 3, benchmark=benchmark)
            self.features['4th_central_moment1'], self.features['4th_central_moment2'] = self.extract_k_central_moment(s_idx, freq, p_idx, 4, benchmark=benchmark)
            self.features['5th_central_moment1'], self.features['5th_central_moment2'] = self.extract_k_central_moment(s_idx, freq, p_idx, 5, benchmark=benchmark)
            self.features['6th_central_moment1'], self.features['6th_central_moment2'] = self.extract_k_central_moment(s_idx, freq, p_idx, 6, benchmark=benchmark)
            self.features['FM41'], self.features['FM42'] = self.extract_FM4(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['median1'], self.features['median2'] = self.extract_median(s_idx, freq, p_idx, benchmark=benchmark)

            # Frequency domain features
            self.features['S11'], self.features['S12'] = self.extract_S1(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S21'], self.features['S22'] = self.extract_S2(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S31'], self.features['S32'] = self.extract_S3(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S41'], self.features['S42'] = self.extract_S4(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S51'], self.features['S52'] = self.extract_S5(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S61'], self.features['S62'] = self.extract_S6(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S71'], self.features['S72'] = self.extract_S7(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S81'], self.features['S82'] = self.extract_S8(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S91'], self.features['S92'] = self.extract_S9(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S101'], self.features['S102'] = self.extract_S10(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S111'], self.features['S112'] = self.extract_S11(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S121'], self.features['S122'] = self.extract_S12(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S131'], self.features['S132'] = self.extract_S13(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S141'], self.features['S142'] = self.extract_S14(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)


            cols = [np.asarray(v) for v in self.features.values()]

            out = np.stack(cols, axis=1)  # (n_states, n_features)
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

            if cache_dir is not None:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                panel_name = Path(self.mat_file).stem
                if benchmark:
                    cache_file = Path(cache_dir) / f"{panel_name}_freq{freq}_path{p_idx}_benchmark_full.npy"
                elif diff:
                    cache_file = Path(cache_dir) / f"{panel_name}_freq{freq}_path{p_idx}_diff_full.npy"
                else:
                    cache_file = Path(cache_dir) / f"{panel_name}_freq{freq}_path{p_idx}_full.npy"
                np.save(str(cache_file), out)
                print(f"Saved extracted features to {cache_file}")

            return out

def pre_compute_features_for_all_panels():
    for freq in range(0, 6):
        for panel_name in BASE_PANELS + PANEL_123_SUBPANELS:
            mat_file = str(mat_file_path(panel_name))
            extractor = FeaturesExtractor(mat_file)
            sampling_rate = 1 / extractor.dt()
            features = extractor.extract_all_features(s_idx=":", freq=freq, p_idx=":", sample_rate=sampling_rate, benchmark=False, diff=True, cache_dir=str(GRAPH_DATA_DIR / "raw"))
            print(f"Extracted features shape for States_{panel_name}.mat: {features.shape}")


if __name__ == "__main__":
    pre_compute_features_for_all_panels()
    