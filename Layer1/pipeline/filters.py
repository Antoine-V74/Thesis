
"""
Low-level ECG signal conditioning utilities.

These functions are intentionally lightweight because the overall project
aims to stay portable to a future Pynapse / real-time implementation.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def butter_bandpass_filter(
    x: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int = 3,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth band-pass filter.

    Parameters
    ----------
    x:
        Input ECG signal.
    fs:
        Sampling frequency in Hz.
    lowcut, highcut:
        Passband edges in Hz.
    order:
        Butterworth filter order.

    Returns
    -------
    np.ndarray
        Filtered ECG.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x)


def notch_filter(
    x: np.ndarray,
    fs: float,
    f0: float = 60.0,
    q: float = 30.0,
) -> np.ndarray:
    """
    Apply a zero-phase notch filter for mains interference.

    If the requested notch frequency is above Nyquist, the function simply
    returns the input unchanged.
    """
    if f0 >= fs / 2:
        return x
    b, a = iirnotch(f0 / (fs / 2), q)
    return filtfilt(b, a, x)
