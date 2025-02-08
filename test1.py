import numpy as np
import cupy as cp
import time

# Simulate a large matrix multiplication for flow test
def simulate_matrix_flow(matrix_size):
    # Generate random matrices (these could be your embeddings)
    matrix_a = cp.random.random((matrix_size, matrix_size)).astype(cp.float32)
    matrix_b = cp.random.random((matrix_size, matrix_size)).astype(cp.float32)

    # Time the matrix multiplication on the GPU
    start_time = time.time()
    result = cp.dot(matrix_a, matrix_b)  # This is the "flow" operation
    cp.cuda.Stream.null.synchronize()  # Synchronize to ensure accurate timing
    end_time = time.time()
    
    return end_time - start_time

# Test logic gates using embeddings for various matrix sizes
def test_gate_operation(matrix_size, gate_type='AND'):
    # Generate random binary matrices for embedding simulation
    matrix_a = cp.random.randint(0, 2, (matrix_size, matrix_size), dtype=cp.int32)
    matrix_b = cp.random.randint(0, 2, (matrix_size, matrix_size), dtype=cp.int32)
    
    # Logic gate simulations using embeddings
    if gate_type == 'AND':
        result = cp.bitwise_and(matrix_a, matrix_b)
    elif gate_type == 'OR':
        result = cp.bitwise_or(matrix_a, matrix_b)
    elif gate_type == 'XOR':
        result = cp.bitwise_xor(matrix_a, matrix_b)
    
    # Timing the gate operation
    start_time = time.time()
    cp.cuda.Stream.null.synchronize()  # Ensure synchronization for accurate timing
    end_time = time.time()
    
    return end_time - start_time

# Main testing function for large matrix flow and gates
def main():
    matrix_sizes = [1024, 2048, 4096, 8192, 16384]  # Increasing matrix sizes
    gate_types = ['AND', 'OR', 'XOR']

    # Matrix flow tests
    for size in matrix_sizes:
        print(f"Matrix size: {size}x{size}")
        
        # Test matrix flow
        time_taken = simulate_matrix_flow(size)
        print(f"Time taken for matrix flow simulation on GPU: {time_taken:.6f} seconds")
        print("-" * 50)
        
        # Test each gate type
        for gate in gate_types:
            time_taken_gate = test_gate_operation(size, gate)
            print(f"Time taken for embedding-based {gate} gate on GPU: {time_taken_gate:.6f} seconds")
            print("-" * 50)

# Run the main testing function
if __name__ == "__main__":
    main()
