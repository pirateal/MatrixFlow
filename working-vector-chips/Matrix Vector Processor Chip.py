import torch
import time

# Matrix Vector Processor Chip (MVPC)
def matrix_vector_multiply(A, B):
    """
    Perform matrix-vector multiplication using the matrix technique.
    A: Matrix (n x m)
    B: Vector (m)
    Returns: Vector (n)
    """
    result = torch.matmul(A, B)
    return result

# Example usage
def mvpc_example():
    # Generate random matrix and vector
    A = torch.randn(64, 64, device="cuda")
    B = torch.randn(64, device="cuda")
    
    start_time = time.perf_counter()
    result = matrix_vector_multiply(A, B)
    end_time = time.perf_counter()
    
    print(f"Matrix-Vector multiplication completed in {end_time - start_time:.4f} seconds.")
    print("Result Vector:", result)

# Run MVPC example
mvpc_example()
