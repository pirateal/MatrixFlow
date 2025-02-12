import torch
import time

# Ensure you are using the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to perform the 8-bit addition and carry propagation
def matrix_8bit_adder(A, B):
    # Move input tensors to GPU
    A = A.to(device)
    B = B.to(device)

    # Start measuring time
    start_time = time.perf_counter()

    # Element-wise addition
    result = torch.add(A, B)  # Perform element-wise addition

    # Carry and sum calculation
    carry = result // 2  # Compute the carry (1 if the sum is 2 or more)
    sum_result = result % 2  # The sum (remainder after division by 2)

    # Time after the operation
    end_time = time.perf_counter()

    # Calculate time in nanoseconds
    elapsed_time_ns = (end_time - start_time) * 1e9

    return sum_result, carry, elapsed_time_ns

# Example 8-bit binary numbers as tensors
A = torch.tensor([1, 0, 1, 1, 0, 0, 1, 1])  # Example operand A (8-bit binary)
B = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1])  # Example operand B (8-bit binary)

# Run the matrix-based adder with precision timing
sum_result, carry_out, elapsed_time_ns = matrix_8bit_adder(A, B)

# Print results
print("Sum:", sum_result)
print("Carry:", carry_out)
print(f"Execution time: {elapsed_time_ns:.2f} ns")

# Optional: Check if it runs in nanoseconds
if elapsed_time_ns < 100:
    print("The operation is in the desired nanosecond range!")
else:
    print("Consider optimizing further for nanosecond precision.")
