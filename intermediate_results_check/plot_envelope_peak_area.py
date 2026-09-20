'''
Diagnostic: plots the Hilbert envelope of one example (panel, state, frequency, path)
scattered signal (amplitude - benchmark_amplitude), marks the envelope peak, and shades
the same "peak lobe" that states.states._envelope_area integrates over (walking outward
from the peak while the envelope keeps decreasing on each side, then
signal_envelope_peak_area scales that trapezoidal sum by dt() for physical time units) --
so the shaded area is exactly what that metric reports, not a guess. Prints both the
locally re-derived area and states.signal_envelope_peak_area's own value as a cross-check.
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
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from states import states
from config import mat_file_path, DEFAULT_FREQ_INDEX

PANEL = "103"
STATE_IDX = 20
FREQ_IDX = DEFAULT_FREQ_INDEX
PATH_IDX = 2


def _envelope_lobe_bounds(envelope):
    '''Same walk as states.states._envelope_area, but returns the (i_start, i_end) index
    bounds of the monotonic lobe around the peak instead of just the area, so the plot
    can shade precisely what that metric integrates over.'''
    peak_idx = int(np.argmax(envelope))

    i_start = peak_idx
    while i_start > 0 and envelope[i_start - 1] <= envelope[i_start] or envelope[i_start - 2] <= envelope[i_start - 1]:
        i_start -= 1

    i_end = peak_idx
    while i_end < len(envelope) - 1 and envelope[i_end + 1] <= envelope[i_end] or envelope[i_end + 2] <= envelope[i_end + 1]:
        i_end += 1

    return i_start, i_end


def main(panel=PANEL, state_idx=STATE_IDX, freq_idx=FREQ_IDX, path_idx=PATH_IDX, save_path=None):
    st = states(str(mat_file_path(panel)))

    time = np.asarray(st.time(state_idx, freq_idx, path_idx))
    last_time = time[-1]
    shifted_time = time[0:len(time)//2]+last_time
    time = np.concatenate((time[0:len(time)//2], shifted_time))
    amp = np.asarray(st.amplitude(state_idx, freq_idx, path_idx))
    bench = np.asarray(st.benchmark_amplitude(state_idx, freq_idx, path_idx))
    scattered = amp - bench

    envelope = np.abs(hilbert(scattered))
    peak_idx = int(np.argmax(envelope))
    i_start, i_end = _envelope_lobe_bounds(envelope)

    # cross-check: the shaded lobe's own trapezoidal area (dx=1, same as _envelope_area's
    # manual sum) times dt() must match states.signal_envelope_peak_area's own value.
    area_local = np.trapz(envelope[i_start:i_end + 1], dx=1.0) * st.dt()
    area_reported = float(st.signal_envelope_peak_area(state_idx, freq_idx, path_idx))
    print(f"Local shaded-area estimate: {area_local:.6g}   "
          f"states.signal_envelope_peak_area: {area_reported:.6g}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(time[100:len(time)//2-1000], scattered[100:len(time)//2-1000], color="lightgray", linewidth=4, label="Scattered signal")
    ax.plot(time[100:len(time)//2-1000], envelope[100:len(time)//2-1000], color="C0", linewidth=5, label="Envelope")
    ax.fill_between(time[i_start:i_end + 1], envelope[i_start:i_end + 1], color="C0", alpha=1.0, zorder=3)
    ax.plot(time[peak_idx], envelope[peak_idx], "o", color="orange", markersize=20, zorder=5, label="Envelope peak")

    ax.set_xlabel("Time", fontsize=50)
    ax.set_ylabel("Amplitude", fontsize=50)
    #ax.legend(fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    save_path = save_path or f"envelope_peak_area_panel{panel}_state{state_idx}_freq{freq_idx}_path{path_idx}_dir1.svg"
    fig.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
