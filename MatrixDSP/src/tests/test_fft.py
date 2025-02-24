# Test cases for FFT computat# tests/test_fft.py
import cupy as cp
from fft_unit import FFTUnit

def test_fft():
    fs = 44100
    fft_unit = FFTUnit(fs)
    t = cp.linspace(0, 1, fs)
    signal = cp.sin(2 * cp.pi * 440 * t)
    spectrum = fft_unit.fft_transform(signal)
    reconstructed = fft_unit.ifft_transform(spectrum)
    error = cp.linalg.norm(signal - cp.real(reconstructed))
    assert error < 1e-5
    print("test_fft passed.")

if __name__ == "__main__":
    test_fft()
ion
