# Test cases for FIR/IIR filtering# tests/test_filter.py
import cupy as cp
from filter_unit import FIRFilter

def test_fir():
    signal = cp.array([1,2,3,4,5,6,7,8,9,10], dtype=cp.float32)
    coeffs = cp.ones(3, dtype=cp.float32) / 3.0
    filtered = FIRFilter(signal, coeffs)
    expected = cp.convolve(signal, coeffs, mode='same')
    error = cp.linalg.norm(filtered - expected)
    assert error < 1e-5
    print("test_fir passed.")

if __name__ == "__main__":
    test_fir()
