import multiprocessing
import time

import numpy as np

from BO_GCN import run_bayesian_optimization
from config import TEST_RUN_DIR, BETA_CONSTANT, FREQUENCY_MAPPING, TYPES, OPTIMIZED_TYPES, OPTIMIZE_RAW
from graph_performance import main as graph_performance_main


freqs = np.arange(len(FREQUENCY_MAPPING))  # frequency indices to sweep over (0-based)


def _beta_suffix(beta):
    return f"_beta{beta}" if beta != BETA_CONSTANT else ""


def _run_one(freq, type, raw_features, beta=BETA_CONSTANT):
    np.random.seed(42)
    run_bayesian_optimization(freq=freq, type=type, raw_features=raw_features, beta=beta)


def _run_in_subprocess(freq, type, raw_features, beta=BETA_CONSTANT):
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_run_one, args=(freq, type, raw_features, beta))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(
            f"Subprocess for type={type!r} freq={freq} raw_features={raw_features} beta={beta} "
            f"failed with exit code {p.exitcode}"
        )

def run_pre_processed_sweep(start_freq=0):
    np.random.seed(42)
    for freq in freqs:
        if freq < start_freq:
            continue
        if not OPTIMIZE_RAW:
            continue
        t_begin = time.perf_counter()
        run_bayesian_optimization(freq=freq, type='peak', raw_features=False, beta=BETA_CONSTANT)
        t_end = time.perf_counter()
        print(f"Time for pre-processed features freq={freq}: {t_end - t_begin:.2f}s\n")

    folders = [f"Bayesian_GCN_peak_freq{freq}" for freq in freqs]
    out_dir = TEST_RUN_DIR / "graph_performance_results_peak"
    graph_performance_main(recompute=True, folders=folders, out_dir=out_dir, raw_features=False)


def run_raw_sweep(start_freq=0):
    np.random.seed(42)
    for freq in freqs:
        if freq < start_freq:
            continue
        if not OPTIMIZE_RAW:
            continue
        t_begin = time.perf_counter()
        _run_in_subprocess(freq, 'peak', True)
        t_end = time.perf_counter()
        print(f"Time for raw features freq={freq}: {t_end - t_begin:.2f}s\n")

    folders = [f"Bayesian_GCN_raw_freq{freq}" for freq in freqs]
    out_dir = TEST_RUN_DIR / "graph_performance_results_raw"
    graph_performance_main(recompute=True, folders=folders, out_dir=out_dir, raw_features=True)


def sweep_over_freq(gcn_type, start_freq=0):
    for freq in freqs:
            if freq < start_freq:
                continue
            t_begin = time.perf_counter()
            run_bayesian_optimization(freq=freq, type=gcn_type, raw_features=True, beta=BETA_CONSTANT)
            t_end = time.perf_counter()
            print(f"Time for {gcn_type} freq={freq}: {t_end - t_begin:.2f}s\n")

def run_types_sweep(start_freq=0):
    """Retrain every GCN type except 'peak' using raw features (peak is left on
    AE-latent features on purpose, see run_raw_sweep)."""
    np.random.seed(42)
    for type in TYPES:
        if type == "peak":
            continue
        if type not in OPTIMIZED_TYPES:
            continue
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(target=sweep_over_freq, args=(type, start_freq))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(
                f"Subprocess for type={type!r} failed with exit code {p.exitcode}"
            )

        folders = [f"Bayesian_GCN_{type}_freq{freq}" for freq in freqs]
        out_dir = TEST_RUN_DIR / f"graph_performance_results_{type}"
        graph_performance_main(recompute=True, folders=folders, out_dir=out_dir, raw_features=True, type=type)


