import torch

# Check available GPUs
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

if num_gpus == 0:
    print("No CUDA devices available. Exiting...")
    exit()

# Function to determine the correct device assignment
def get_device(chip_id):
    return f'cuda:{chip_id % num_gpus}'

# CUDA-based matrix multiplication with GPU timing
def matrix_multiplication(matrix_a, matrix_b, stream):
    with torch.cuda.stream(stream):
        return torch.matmul(matrix_a, matrix_b)

# CUDA-based memory operation with GPU timing
def memory_operation(matrix, stream):
    with torch.cuda.stream(stream):
        return torch.sum(matrix)

# Define the core execution task with nanosecond timing
def core_task(core_id, task_name, matrix_a, matrix_b=None, stream=None):
    device = get_device(core_id)
    stream = torch.cuda.Stream(device) if stream is None else stream

    matrix_a = matrix_a.to(device, non_blocking=True)
    if matrix_b is not None:
        matrix_b = matrix_b.to(device, non_blocking=True)

    # Synchronize before starting timing
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record(stream)
    if task_name == "matrix_multiplication":
        result = matrix_multiplication(matrix_a, matrix_b, stream)
    elif task_name == "memory_operation":
        result = memory_operation(matrix_a, stream)
    else:
        raise ValueError("Unknown task")
    end_event.record(stream)

    # Synchronize to ensure accurate timing
    torch.cuda.synchronize(device)

    elapsed_time_ns = start_event.elapsed_time(end_event) * 1_000_000  # Convert ms to ns
    print(f"[Core {core_id} on {device}] {task_name} completed in {elapsed_time_ns:.0f} ns")

# Run a single chip simulation using CUDA streams
def run_chip_simulation(chip_id):
    device = get_device(chip_id)
    print(f"Running Chip {chip_id} Simulation on {device}")

    matrix_a = torch.randn((128, 128), device=device)
    matrix_b = torch.randn((128, 128), device=device)

    streams = [torch.cuda.Stream(device) for _ in range(8)]

    for core_id in range(8):  # 8 cores per chip
        task_name = "matrix_multiplication" if core_id % 2 == 0 else "memory_operation"
        core_task(core_id, task_name, matrix_a, matrix_b if task_name == "matrix_multiplication" else None, streams[core_id])

    torch.cuda.synchronize(device)
    print(f"Chip {chip_id} simulation completed")

# Run all 8 chips in sequence (still inside the GPU)
def run_system_simulation():
    num_chips = 8
    for chip_id in range(num_chips):
        run_chip_simulation(chip_id)

    print("System simulation completed")

# Run the system
if __name__ == "__main__":
    run_system_simulation()
