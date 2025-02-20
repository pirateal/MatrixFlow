import cupy as cp
import time

# Define memory chip size (1GB per chip, 1024x1024 matrix with 1-bit cells)
chip_size = (1024, 1024, 1)  # 1GB per chip, using a 1024x1024 matrix with 1-bit cells
num_chips = 8192  # Total of 8192 chips for 8TB of memory
total_memory_size = chip_size[0] * chip_size[1] * num_chips  # Total size for 8TB

# Initialize the 8192 chips (8TB memory, each chip 1GB) using CuPy for GPU
ram_chips = cp.zeros((num_chips, *chip_size), dtype=cp.uint8)

# Split the memory into two halves (4TB each)
half_num_chips = num_chips // 2  # 4096 chips per half for 4TB

# Optimized function for matrix-based data movement
def move_data_logically_optimized(src_chip, dest_chip):
    # Perform a matrix-wise copy using slice assignments (fully utilizing GPU parallelism)
    dest_chip[:] = src_chip  # Efficient memory block copy (GPU parallelized)
    
# Test data throughput (moving data from one half to another)
def move_data_between_halves_optimized():
    start_time = time.time()

    # Move data from the first half to the second half (using parallelized matrix operations)
    for i in range(half_num_chips):
        move_data_logically_optimized(ram_chips[i], ram_chips[half_num_chips + i])  # Copy from first half to second half

    # Move data back from the second half to the first half (using parallelized matrix operations)
    for i in range(half_num_chips):
        move_data_logically_optimized(ram_chips[half_num_chips + i], ram_chips[i])  # Copy back to the first half

    # Calculate throughput
    elapsed_time = time.time() - start_time
    throughput = total_memory_size / elapsed_time  # Bytes per second

    print(f"Time for full data move back and forth between halves: {elapsed_time:.4f} seconds")
    print(f"Data throughput: {throughput / (1024 ** 3):.4f} GB/s")

# Run the optimized data movement test
print("Starting optimized data movement between halves of 8TB memory (through matrix logic)...")
move_data_between_halves_optimized()
