import sys
from pathlib import Path
import pickle

import numpy as np
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "tools", "training", "intermediate_results_check", "results_analysis"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from graph_performance import PANELS, average_HI_over_frequency
from Fitness_summary import METRIC_NAMES
from prognostic_criteria import monotonicity_criterion, trendability_criterion, prognosability_criterion
from config import  FREQUENCY_MAPPING, GRAPH_TYPES, TEST_PANEL, TEST_RUN_DIR, FOLD_KEYS


def main():
    types = GRAPH_TYPES + ["raw"]
    model_dirs = [TEST_RUN_DIR / f"graph_performance_results_{t}" for t in types]
    model_dirs.append(TEST_RUN_DIR / "path_performance_results")


    for model_dir in model_dirs:
        
        HI = np.load(model_dir / "HI.pkl", allow_pickle=True)  # (folds, freq, panels)

        HI_test_metrics = np.zeros((len(HI),len(HI[0])+1, 4))  # (fold, freq, metric)
        test_panel_idx = PANELS.index(int(TEST_PANEL[0]))  # position of config.TEST_PANEL within PANELS

        for fold in range(len(HI)):
            for freq in range(len(HI[0])+1):  # include average frequency
                if freq == len(HI[0]):  # average frequency
                    fold_key = FOLD_KEYS[fold]
                    avg_curves = [average_HI_over_frequency(HI, panel, fold=fold_key) for panel in PANELS]
                    avg_curves = [c for c in avg_curves if c is not None]
                    if not avg_curves:
                        continue
                    Mo = monotonicity_criterion(avg_curves, test_mode=True, test_idx=test_panel_idx)
                    Tr = trendability_criterion(avg_curves)
                    Pr = prognosability_criterion(avg_curves, test_mode=True, test_idx=test_panel_idx)
                    HI_test_metrics[fold][freq] = [Mo + Tr + Pr, Mo, Pr, Tr]
                    continue
                Mo = monotonicity_criterion(HI[fold][freq], test_mode=True, test_idx=test_panel_idx)
                Pr = prognosability_criterion(HI[fold][freq], test_mode=True, test_idx=test_panel_idx)
                Tr = trendability_criterion(HI[fold][freq]) # trendability does not support test_mode
                fitness = Mo + Pr + Tr
                HI_test_metrics[fold][freq] = [fitness, Mo, Pr, Tr]

        # write to HI_test_metrics.pkl
        with open(model_dir / "HI_test_metrics.pkl", "wb") as f:
            pickle.dump(HI_test_metrics, f)

        #remove the HI_test_metrics.pkl.npy file if it exists
        npy_path = model_dir / "HI_test_metrics.pkl.npy"
        if npy_path.exists():
            npy_path.unlink()

        # save to HI_metrics.xlsx as a new sheet
        import pandas as pd
        xlsx_path = model_dir / "HI_metrics.xlsx"

        freq_labels = [f"{FREQUENCY_MAPPING[i]} kHz" for i in range(len(FREQUENCY_MAPPING))] + ["average"]
        rows = []
        for fold_idx, fold in enumerate(FOLD_KEYS):
            for freq_idx, freq_label in enumerate(freq_labels):
                row = {"fold": FOLD_KEYS[fold_idx], "frequency": freq_label}
                row.update(zip(METRIC_NAMES, HI_test_metrics[fold_idx, freq_idx]))
                rows.append(row)
        df = pd.DataFrame(rows)
        if not xlsx_path.exists():
            raise FileNotFoundError(f"Expected {xlsx_path} to exist. Run graph_performance.py first.")
        with pd.ExcelWriter(xlsx_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name="test_HI_metrics", index=False)

if __name__ == "__main__":
    main()

        



        