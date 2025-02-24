# examples/test_audio_filter.py
import cupy as cp
import numpy as np
import sounddevice as sd
from src.dsp_core import DSPCore

def main():
    fs = 44100
    dsp = DSPCore(fs)
    t = cp.linspace(0, 1, fs)
    # Generate a 440 Hz sine wave signal
    signal = cp.sin(2 * cp.pi * 440 * t)
    # FIR filter coefficients (simple moving average)
    coeffs = cp.ones(5) / 5.0
    output = dsp.process_signal(signal, coeffs)
    # Play the processed signal
    sd.play(cp.asnumpy(cp.real(output)), fs)
    sd.wait()

if __name__ == "__main__":
    main()
# Example script to apply DSP filters to audio
