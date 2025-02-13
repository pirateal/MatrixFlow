import torch
import time

# Ensure we're using CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################
# Matrix Processor Chip (MPC) - Handles large-scale matrix operations
##########################################
class MatrixProcessorChip:
    def __init__(self, matrix_size=64):
        self.matrix_size = matrix_size
        self.A = torch.randn(self.matrix_size, self.matrix_size, device=device)
        self.B = torch.randn(self.matrix_size, self.matrix_size, device=device)
        self.result = torch.zeros(self.matrix_size, self.matrix_size, device=device)
    
    def matrix_multiply(self):
        start_time = time.perf_counter()
        self.result = torch.matmul(self.A, self.B)
        end_time = time.perf_counter()
        return end_time - start_time
    
    def matrix_addition(self):
        start_time = time.perf_counter()
        self.result = self.A + self.B
        end_time = time.perf_counter()
        return end_time - start_time
    
    def matrix_transpose(self):
        start_time = time.perf_counter()
        self.result = self.A.T
        end_time = time.perf_counter()
        return end_time - start_time
    
    def print_result(self):
        print("Result Matrix:")
        print(self.result)

# Initialize the Matrix Processor Chip
mpc = MatrixProcessorChip(matrix_size=64)

# Perform Matrix Multiplication
mul_time = mpc.matrix_multiply()
print(f"Matrix multiplication completed in {mul_time:.4f} seconds.")

# Perform Matrix Addition
add_time = mpc.matrix_addition()
print(f"Matrix addition completed in {add_time:.4f} seconds.")

# Perform Matrix Transposition
transpose_time = mpc.matrix_transpose()
print(f"Matrix transposition completed in {transpose_time:.4f} seconds.")

# Print the result matrix of the last operation
mpc.print_result()
