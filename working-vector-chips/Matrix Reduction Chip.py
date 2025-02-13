import torch
import time

# Function to perform matrix reduction (sum across rows or columns)
def matrix_reduction(matrix, reduction_type="sum", axis=0):
    """
    Perform reduction operation (sum or max) on a matrix.
    :param matrix: Input matrix
    :param reduction_type: 'sum' or 'max' (default is 'sum')
    :param axis: Axis along which to reduce (0 for rows, 1 for columns)
    :return: Reduced matrix
    """
    if reduction_type == "sum":
        return torch.sum(matrix, dim=axis)
    elif reduction_type == "max":
        return torch.max(matrix, dim=axis)[0]  # Only take the max value, not indices
    else:
        raise ValueError("Unsupported reduction type. Use 'sum' or 'max'.")

# Main chip script to execute matrix reduction
def matrix_reduction_chip():
    # Generate a random matrix (e.g., 64x64)
    matrix = torch.randn(64, 64, device="cuda")
    print(f"Original Matrix:\n{matrix}\n")

    # Perform matrix reduction (sum) across rows (axis=0)
    start_time = time.perf_counter()
    reduced_matrix_sum = matrix_reduction(matrix, reduction_type="sum", axis=0)
    end_time = time.perf_counter()
    print(f"Matrix Reduction (sum across rows) completed in {end_time - start_time:.4f} seconds.")
    print(f"Reduced Matrix (sum across rows):\n{reduced_matrix_sum}\n")

    # Perform matrix reduction (max) across columns (axis=1)
    start_time = time.perf_counter()
    reduced_matrix_max = matrix_reduction(matrix, reduction_type="max", axis=1)
    end_time = time.perf_counter()
    print(f"Matrix Reduction (max across columns) completed in {end_time - start_time:.4f} seconds.")
    print(f"Reduced Matrix (max across columns):\n{reduced_matrix_max}\n")

# Run the Matrix Reduction Chip
matrix_reduction_chip()
