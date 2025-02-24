# examples/test_fft_analysis.py
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from src.fft_unit import FFTUnit

def main():
    fs = 44100
    fft_unit = FFTUnit(fs)
    t = cp.linspace(0, 1, fs)
    signal = cp.sin(2 * cp.pi * 440 * t)
    spectrum = fft_unit.fft_transform(signal)
    freqs = cp.fft.fftfreq(signal.shape[0], 1/fs)
    plt.figure()
    plt.plot(cp.asnumpy(freqs), cp.asnumpy(cp.abs(spectrum)))
    plt.title("FFT Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, fs/2)
    plt.show()

if __name__ == "__main__":
    main()
