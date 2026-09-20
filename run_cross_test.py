'''
This file runs "run_single_panel_serial.py" script for test panel specified in PANELS.
New process is spawned for each panel. 
Initialy used for BO sweep, so there is also an option to specify the starting frequency index for each panel (START_FREQ).
'''

import subprocess
import sys
import time
from pathlib import Path

PANELS = [ "103", "104", "105", "109"]
START_FREQ = {"104": 0}  # panel -> freq to resume from (freq 1-3 already ran for 104); defaults to 1
_WORKER = Path(__file__).resolve().parent / "training" / "run_single_panel_serial.py"

for panel in PANELS:
    start_freq = START_FREQ.get(panel, 0)
    print(f"=== Starting raw BO sweep for TEST_PANEL={panel} (from freq={start_freq}) ===")
    t_begin = time.perf_counter()
    result = subprocess.run([sys.executable, str(_WORKER), panel, str(start_freq)])
    t_end = time.perf_counter()
    print(f"=== Finished TEST_PANEL={panel} in {t_end - t_begin:.2f}s (exit code {result.returncode}) ===\n")
    if result.returncode != 0:
        raise RuntimeError(f"Raw BO sweep failed for TEST_PANEL={panel} (exit code {result.returncode})")
