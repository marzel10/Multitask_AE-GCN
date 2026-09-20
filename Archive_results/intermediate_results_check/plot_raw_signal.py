import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "tools", "training", "intermediate_results_check", "results_analysis"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
from states import states
from config import mat_file_path, DEFAULT_FREQ_INDEX

# Load panels once
st_103 = states(str(mat_file_path("103")))
st_104 = states(str(mat_file_path("104")))
st_105 = states(str(mat_file_path("105")))
st_109 = states(str(mat_file_path("109")))


state_idx = [0, 11, 20, 26] # indices of states to plot (0-based)
freq = DEFAULT_FREQ_INDEX   # frequency index to plot (0-based)
p_idx = 2                   # index of path to plot (0-based). path is represented as a pair of transducers 

panels = [
    ("103", st_103),
    ("104", st_104),
    ("105", st_105),
    ("109", st_109),
]

for name, state in panels:
    print(f"{name} - Total states: {state.num_states}, Frequencies: {state.num_freq}, Pair indices: {state.num_pair}")

for st_idx in state_idx:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    axes = axes.ravel()

    for ax, (name, st_obj) in zip(axes, panels):
        time = st_obj.time(st_idx, freq, p_idx)
        amp = st_obj.amplitude(st_idx, freq, p_idx)
        bench = st_obj.benchmark_amplitude(st_idx, freq, p_idx)
        scattered = amp-bench
        ax.plot(time, scattered, label=f"State {st_idx} Panel {name}")
        ax.set_title(f"Panel {name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.legend()

    fig.suptitle(f"Signals for state {st_idx} (freq {freq}, pair {p_idx})")
    fig.tight_layout()
    plt.show()

st_123_43 = states(str(mat_file_path("123_43")))
st_123_43.plot(1, DEFAULT_FREQ_INDEX, 14, direction=0, save_path="state_123_41_dir1.svg")
st_123_43.plot(1, DEFAULT_FREQ_INDEX, 14, direction=1, save_path="state_123_41_dir2.svg")