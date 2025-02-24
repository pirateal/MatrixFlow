# modulation_unit.py
import cupy as cp

def am_modulate(signal, carrier_freq, fs=44100, modulation_index=0.5):
    """
    Amplitude modulate the input signal.
    
    signal: cp.array, input signal.
    carrier_freq: float, carrier frequency in Hz.
    modulation_index: float, modulation depth (0 to 1).
    Returns: cp.array, AM modulated signal.
    """
    t = cp.arange(signal.shape[0]) / fs
    carrier = cp.cos(2 * cp.pi * carrier_freq * t)
    return (1 + modulation_index * signal) * carrier

def fm_modulate(signal, carrier_freq, fs=44100, modulation_index=50.0):
    """
    Frequency modulate the input signal.
    
    signal: cp.array, input signal (assumed normalized between -1 and 1).
    carrier_freq: float, carrier frequency in Hz.
    modulation_index: float, frequency deviation.
    Returns: cp.array, FM modulated signal.
    """
    t = cp.arange(signal.shape[0]) / fs
    # Integrate the signal (cumulative sum approximates integration)
    integral = cp.cumsum(signal) / fs
    return cp.cos(2 * cp.pi * carrier_freq * t + 2 * cp.pi * modulation_index * integral)
