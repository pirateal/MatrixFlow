# Test cases for the DSP core# tests/test_dsp.py
import cupy as cp
import numpy as np
import sys
sys.path.append("../src")
from dsp_core import DSPCore

def test_process_signal():
    fs = 44100
    dsp = DSPCore(fs)
    # Create a dummy sine wave signal
    t = cp.linspace(0, 1, fs)
    signal = cp.sin(2 * cp.pi * 440 * t)
    # Define simple FIR filter coefficients (moving average)
    coeffs = cp.ones(5) / 5.0
    output = dsp.process_signal(signal, coeffs)
    assert output.shape == signal.shape
    print("test_process_signal passed.")

if __name__ == "__main__":
    test_process_signal()
