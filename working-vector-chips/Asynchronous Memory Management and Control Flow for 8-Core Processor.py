import torch
import time
import threading

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")

# Set the device to GPU (using the first available GPU)
torch.cuda.set_device(0)  # Use device 0 (your first GPU)

# Define the matrix operation functions
def matrix_multiplication(A, B):
    device = A.device  # Ensure the matrix device is used
    return torch.matmul(A, B)

def matrix_addition(A, B):
    device = A.device  # Ensure the matrix device is used
    return torch.add(A, B)

def matrix_reduction_sum(A):
    device = A.device  # Ensure the matrix device is used
    return torch.sum(A, dim=1)

# Create a function to simulate memory operations asynchronously
def memory_operation(matrix):
    device = matrix.device  # Ensure the matrix device is used
    result = matrix_addition(matrix, matrix)  # Example of memory operation
    return result

# Create a function to simulate a core executing a task
def core_task(core_id, task, *args):
    print(f"[Core {core_id}] Executing task: {task.__name__}")
    
    # Set CUDA device per thread
    device = args[0].device  # Use the device of the first argument
    torch.cuda.set_device(device)  # Set the device for this thread
    
    # Synchronize CUDA to ensure context is properly set before running the task
    torch.cuda.synchronize()

    start_time = time.time()
    result = task(*args)
    end_time = time.time()
    print(f"[Core {core_id}] Task {task.__name__} completed in {end_time - start_time:.4f} seconds")
    return result

# Define a function to run the processor simulation
def run_processor_simulation():
    # Set up the matrices for computation
    matrix_a = torch.randn((128, 128), device='cuda')
    matrix_b = torch.randn((128, 128), device='cuda')

    # Synchronize CUDA in the main thread before starting
    torch.cuda.synchronize()

    # Create 8 threads to simulate 8 cores
    threads = []
    results = []

    # Example tasks for each core to run in parallel
    for core_id in range(8):
        if core_id % 2 == 0:
            # Assign Matrix Multiplication to even cores
            thread = threading.Thread(target=core_task, args=(core_id, matrix_multiplication, matrix_a, matrix_b))
        else:
            # Assign Memory Operations to odd cores
            thread = threading.Thread(target=core_task, args=(core_id, memory_operation, matrix_a))
        
        threads.append(thread)

    # Start all the threads (cores)
    for thread in threads:
        thread.start()

    # Wait for all threads to complete (synchronization point)
    for thread in threads:
        thread.join()

    print("Processor simulation completed")

# Run the processor simulation
run_processor_simulation()
