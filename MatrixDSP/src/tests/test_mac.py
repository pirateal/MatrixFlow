# tests/test_mac.py
import cupy as cp
from matrix_mac import matrix_mac

def test_matrix_mac():
    A = cp.array([[1,2],[3,4]], dtype=cp.float32)
    B = cp.array([[5,6],[7,8]], dtype=cp.float32)
    bias = cp.array([1,1], dtype=cp.float32)
    result = matrix_mac(A, B, bias)
    expected = cp.array([[20,23],[44,51]], dtype=cp.float32)
    error = cp.linalg.norm(result - expected)
    assert error < 1e-5
    print("test_matrix_mac passed.")

if __name__ == "__main__":
    test_matrix_mac()
