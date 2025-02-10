import torch
import time

# Detect GPU or fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **Batch Size & Parallel Configurations**
BATCH_SIZES = [4096, 8192, 16384]  # Test different batch sizes
ITERATIONS = 5000  # Total iterations per test

# Optimized MatrixFlow Scratchpad
scratchpad = torch.rand((80, 256), dtype=torch.float32, device=device)

# **Define High-Performance Parallelized MatrixFlow Hash Functions**
def matrixflow_parallel(batch_size):
    """Parallelized MatrixFlow Hashing"""
    input_data = torch.rand((batch_size, 80), dtype=torch.float32, device=device)

    # Use CUDA's Autocast (Automatic Mixed Precision) for GPU speed-up
    with torch.amp.autocast("cuda"):
        temp = torch.sin(torch.matmul(input_data, scratchpad))  # Matrix multiplication
        temp = torch.cos(temp) * torch.tanh(temp)  # Sin/Cos-based transformations
    
    return temp.sum()  # Reduce output to single hash result

# **Benchmark Function for Parallel Execution**
def benchmark_parallel(batch_size):
    print(f"🚀 Running Parallel MatrixFlow Benchmark with Batch Size: {batch_size}...\n")

    start_time = time.time()
    total_hashes = 0

    for i in range(ITERATIONS):
        _ = matrixflow_parallel(batch_size)  # Run hash function
        total_hashes += batch_size

        elapsed_time = time.time() - start_time
        if elapsed_time == 0:
            elapsed_time = 1e-6  # Prevent zero division

        if i % 1000 == 0:
            hash_rate = total_hashes / elapsed_time
            print(f"[BATCH {batch_size}] Iteration {i}/{ITERATIONS} | Hash Rate: {hash_rate:.2f} H/s")

    total_time = time.time() - start_time
    if total_time == 0:
        total_time = 1e-6  # Prevent zero division
    
    final_hash_rate = total_hashes / total_time
    print(f"✅ [FINAL] Batch Size {batch_size} Hash Rate: {final_hash_rate:.2f} H/s\n")

if __name__ == "__main__":
    print("🚀 Running High-Performance MatrixFlow Parallel Tests...\n")

    for batch_size in BATCH_SIZES:
        benchmark_parallel(batch_size)
