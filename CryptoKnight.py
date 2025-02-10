import torch
import time
import hashlib
import numpy as np
from torch import nn
from torch.autograd import Variable

# Define your GPU settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define MatrixFlow Hashing Simulation
class MatrixFlowHashing:
    def __init__(self, batch_size=4096, iterations=5000):
        self.batch_size = batch_size
        self.iterations = iterations
        self.lookup_table = None  # Lookup table for precomputed hashes
        self.input_data = None  # Input data for hashing
    
    def initialize(self):
        """
        Initializes the input data and lookup table
        """
        self.input_data = torch.randint(low=0, high=255, size=(self.batch_size, 256), dtype=torch.uint8, device=device)
        self.lookup_table = torch.rand((self.batch_size, 256), dtype=torch.float32, device=device)
    
    def hash_function(self, input_data):
        """
        Simulated CryptoNight hash function using a simple transformation
        """
        # You can modify the hash function as needed to simulate CryptoNight
        hash_data = input_data.float().sum(dim=1)
        hash_data = hash_data % 256  # Simple modulo operation to simulate hashing
        return hash_data

    def run(self):
        """
        Main function to run the parallel MatrixFlow benchmark
        """
        self.initialize()
        for i in range(self.iterations):
            start_time = time.perf_counter()  # Use perf_counter() for higher precision
            # Simulate MatrixFlow Hashing using the lookup table and GPU operations
            output = self.hash_function(self.input_data)  # Apply simulated hash function
            # Capture hash rate
            elapsed_time = time.perf_counter() - start_time
            if elapsed_time > 0:  # Avoid division by zero
                hash_rate = self.batch_size / elapsed_time
                print(f"[BATCH {self.batch_size}] Iteration {i}/{self.iterations} | Hash Rate: {hash_rate:.2f} H/s")
            else:
                print(f"[BATCH {self.batch_size}] Iteration {i}/{self.iterations} | Elapsed time was too small to measure")
        print(f"✅ Final Hash Rate for Batch {self.batch_size}: {hash_rate:.2f} H/s")


# Main function to start the benchmarking
def run_benchmark():
    # List of batch sizes to test
    batch_sizes = [4096, 8192, 16384]

    for batch_size in batch_sizes:
        print(f"🚀 Running Parallel MatrixFlow Benchmark with Batch Size: {batch_size}...")
        benchmark = MatrixFlowHashing(batch_size=batch_size, iterations=5000)
        benchmark.run()


# Run the benchmark
if __name__ == '__main__':
    run_benchmark()
