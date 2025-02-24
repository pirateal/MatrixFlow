# matrix_mac.py
import cupy as cp

def matrix_mac(input_matrix, weight_matrix, bias=None):
    """
    Perform a multiply-accumulate (MAC) operation on matrices.
    
    input_matrix: cp.array of shape (M, N)
    weight_matrix: cp.array of shape (N, P)
    bias: cp.array of shape (P,), optional
    Returns: cp.array of shape (M, P)
    """
    result = cp.dot(input_matrix, weight_matrix)
    if bias is not None:
        result += bias
    return result
