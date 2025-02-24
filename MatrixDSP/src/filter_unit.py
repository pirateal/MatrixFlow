# filter_unit.py
import cupy as cp

def FIRFilter(signal, coeffs):
    """
    Apply an FIR filter to the input signal using convolution.
    
    signal: cp.array, 1D input signal.
    coeffs: cp.array, filter coefficients.
    Returns: cp.array, filtered signal.
    """
    return cp.convolve(signal, coeffs, mode='same')

def IIRFilter(signal, coeffs_a, coeffs_b):
    """
    Apply an IIR filter to the input signal using a recursive implementation.
    
    signal: cp.array, 1D input signal.
    coeffs_a: cp.array, denominator coefficients (with a[0] assumed 1).
    coeffs_b: cp.array, numerator coefficients.
    Returns: cp.array, filtered signal.
    """
    N = signal.shape[0]
    filtered = cp.zeros_like(signal)
    order = len(coeffs_a)
    for n in range(N):
        acc = 0
        for i in range(len(coeffs_b)):
            if n - i >= 0:
                acc += coeffs_b[i] * signal[n - i]
        for j in range(1, order):
            if n - j >= 0:
                acc -= coeffs_a[j] * filtered[n - j]
        filtered[n] = acc
    return filtered
