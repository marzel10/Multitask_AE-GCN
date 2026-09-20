'''This script computes the Weighted Average Ensemble (WAE) Health Index (HI) and its associated metrics for each model directory.
 It loads the HI metrics and test metrics from the specified model directories, computes the WAE HI and its metrics, and saves the results to WAE_HI_metrics.pkl.
 Additionally, it appends the WAE HI metrics to an existing Excel file containing the HI metrics.'''
import pickle
import sys
from pathlib import Path

import numpy as np
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "tools", "training", "intermediate_results_check", "results_analysis"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from prognostic_criteria import monotonicity_criterion, trendability_criterion, prognosability_criterion
from config import GRAPH_TYPES, FOLD_KEYS, TEST_RUN_DIR

def main():
    types = GRAPH_TYPES + ["raw"]
    model_dirs = [TEST_RUN_DIR / f"graph_performance_results_{t}" for t in types]
    model_dirs.append(TEST_RUN_DIR / "path_performance_results")

    for model_dir in model_dirs:
        if not model_dir.exists():
            raise FileNotFoundError(f"Expected {model_dir} to exist. Run sweep_over_options.py first.")

        # load the HI_metrics.pkl file from each model_dir
        HI_metrics = np.load(model_dir / "HI_metrics.pkl", allow_pickle=True)  # (folds, freq, metric)
        HI_test_metrics = np.load(model_dir / "HI_test_metrics.pkl", allow_pickle=True)  # (folds, freq, metric)
        HI = np.load(model_dir / "HI.pkl", allow_pickle=True)  # 
        print(f"Loaded metrics: {HI_metrics.shape[0]} folds x {HI_metrics.shape[1]} freqs x {HI_metrics.shape[2]} metrics, dtype {HI_metrics.dtype}")
        print(f"Loaded HI: {len(HI)} folds x {len(HI[0])} freqs x {len(HI[0][0])} panels, "
            f"leaf shapes e.g. {[HI[0][0][p].shape if HI[0][0][p] is not None else None for p in range(len(HI[0][0]))]}")

        fitness = HI_metrics[:, :, 0]  # (folds, freq_total)
        WAE_weights = fitness[:, :-1] / np.sum(fitness[:, :-1], axis=1, keepdims=True)  # (folds, freq_total - 1)
        
        WAE_HI = np.zeros((len(HI),len(HI[0][0])), dtype=object)  # (folds, panels)
        WAE_HI_metrics = np.zeros((len(HI),HI_metrics.shape[2]), dtype=float)  # (folds, panels, metrics)
        WAE_HI_test_metrics = np.zeros((len(HI),HI_metrics.shape[2]), dtype=float)  # (folds, panels, metrics)

        for fold in range(len(HI)):
            for panel in range(len(HI[fold][0])):
                for freq in range(len(HI[0])-1):
                    if HI[fold][freq][panel] is not None:
                        if WAE_HI[fold][panel] is None:
                            WAE_HI[fold][panel] = np.zeros_like(HI[fold][freq][panel])
                        WAE_HI[fold][panel] += WAE_weights[fold][freq] * HI[fold][freq][panel]
                        
            Mo = monotonicity_criterion(WAE_HI[fold])
            Pr = prognosability_criterion(WAE_HI[fold])
            Tr = trendability_criterion(WAE_HI[fold])
            fitness = Mo + Pr + Tr
            WAE_HI_metrics[fold] = [fitness, Mo, Pr, Tr]

            # compute test metrics for WAE_HI
            l1_23_idx = len(WAE_HI[fold]) - 1
            Mo_test = monotonicity_criterion(WAE_HI[fold], test_mode=True, test_idx=l1_23_idx)
            Pr_test = prognosability_criterion(WAE_HI[fold], test_mode=True, test_idx=l1_23_idx)
            Tr_test = trendability_criterion(WAE_HI[fold])  # trendability does not support test_mode
            fitness_test = Mo_test + Pr_test + Tr_test
            WAE_HI_test_metrics[fold] = [fitness_test, Mo_test, Pr_test, Tr_test]
            
        print(f"Computed WAE_HI: {len(WAE_HI)} folds x {len(WAE_HI[0])} panels, "
            f"leaf shapes e.g. {[WAE_HI[0][p].shape if WAE_HI[0][p] is not None else None for p in range(len(WAE_HI[0]))]}")

        print(f"Computed WAE_HI_metrics: {WAE_HI_metrics.shape[0]} folds x  {WAE_HI_metrics.shape[1]} metrics, dtype {WAE_HI_metrics.dtype}")

        with open(model_dir / "WAE_HI_metrics.pkl", "wb") as f:
            pickle.dump(WAE_HI_metrics, f)
        print(f"Saved: {model_dir / 'WAE_HI_metrics.pkl'}")

        with open(model_dir / "WAE_HI_test_metrics.pkl", "wb") as f:
            pickle.dump(WAE_HI_test_metrics, f)
        print(f"Saved: {model_dir / 'WAE_HI_test_metrics.pkl'}")

        # open HI_metrics.xlsx and append a new sheet with the WAE_HI_metrics
        import pandas as pd

        xlsx_path = model_dir / "HI_metrics.xlsx"
        
        if not xlsx_path.exists():
            raise FileNotFoundError(f"Expected {xlsx_path} to exist. Run sweep_over_options.py first.")

        with pd.ExcelWriter(xlsx_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            # create a DataFrame for WAE_HI_metrics
            df = pd.DataFrame(WAE_HI_metrics, columns=["fitness", "monotonicity", "prognosability", "trendability"])
            df.index = FOLD_KEYS
            df.index.name = "fold"
            df.to_excel(writer, sheet_name="WAE_HI_metrics")

            df = pd.DataFrame(WAE_HI_test_metrics, columns=["fitness", "monotonicity", "prognosability", "trendability"])
            df.index = FOLD_KEYS
            df.index.name = "fold"
            df.to_excel(writer, sheet_name="WAE_HI_test_metrics")

if __name__ == "__main__":
    main()

