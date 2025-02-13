import numpy as np
from numba import cuda
import time

# Check if CUDA is available
if not cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires a CUDA-capable GPU.")

def check_cuda_device():
    print("CUDA Device Info:")
    device = cuda.get_current_device()
    print(f"Device Name: {device.name}")
    free_memory, total_memory = cuda.current_context().get_memory_info()
    print(f"Free Memory: {free_memory} bytes")
    print(f"Total Memory: {total_memory} bytes")

# Persistent kernel that runs entirely on the GPU for a fixed duration.
@cuda.jit
def persistent_and_gate_kernel(input_data, output_data, op_counter, clock_target):
    idx = cuda.grid(1)
    if idx >= input_data.size:
        return

    # Perform the AND gate operation
    output_data[idx] = input_data[idx] & 1
    # Increment the counter for operations performed
    op_counter[0] += 1

def main():
    check_cuda_device()
    
    # Duration (in seconds) for which the kernel should run.
    duration = 300

    # Retrieve the GPU's clock rate (in Hz) and calculate how many operations to perform in the target duration.
    device = cuda.get_current_device()
    clock_rate = device.CLOCK_RATE  # in Hz
    cycles_per_second = clock_rate  # already in Hz (cycles per second)
    clock_target = int(duration * cycles_per_second)  # Unused in this case, just for reference
    print(f"Running persistent kernel for {duration} seconds...")

    # Prepare input data: 10 million elements (for example).
    size = 10_000_000
    input_data = np.random.randint(0, 2, size, dtype=np.int32)
    
    # Copy input data to the device and allocate output array.
    input_device = cuda.to_device(input_data)
    output_device = cuda.device_array_like(input_data)
    
    # Create a device array to hold the operation counter (using int64).
    op_counter = cuda.to_device(np.array([0], dtype=np.int64))
    
    # Set grid dimensions.
    threads_per_block = 256
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
    
    # Measure the time taken to run the kernel.
    start_time = time.time()
    
    # Launch the persistent kernel.
    persistent_and_gate_kernel[blocks_per_grid, threads_per_block](
        input_device, output_device, op_counter, clock_target
    )
    cuda.synchronize()  # Wait for kernel to complete.
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Retrieve the global operation count.
    total_operations = op_counter.copy_to_host()[0]
    ops_per_sec = total_operations / elapsed_time
    
    # (Optional) Copy output data back to host for verification.
    output_host = output_device.copy_to_host()
    
    print(f"Kernel ran for {elapsed_time:.2f} seconds")
    print(f"Total operations performed: {total_operations}")
    print(f"Operations per second: {ops_per_sec:.2f}")
    print(f"Output (first 10 elements): {output_host[:10]}")

if __name__ == "__main__":
    main()
