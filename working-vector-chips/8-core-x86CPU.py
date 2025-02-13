import torch
import time
import queue
import random
import concurrent.futures

# Ensure we use CUDA if available (your matrix technique leverages GPU matrix operations)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################
# ALU Functions (Vectorized) using Matrix Technique
##########################################
def alu_64bit_vectorized(A, B, operation="add"):
    if operation == "add":
        result = A + B
        carry = (result > 1).to(torch.uint8)  # Carry: where sum exceeds 1
        result = result % 2                   # Ensure binary result (0 or 1)
        return result, carry
    elif operation == "subtract":
        result = A - B
        carry = (result < 0).to(torch.uint8)  # Borrow in subtraction
        result = result % 2
        return result, carry
    elif operation == "and":
        return A & B, torch.tensor(0, dtype=torch.uint8, device=device)
    elif operation == "or":
        return A | B, torch.tensor(0, dtype=torch.uint8, device=device)
    elif operation == "xor":
        return A ^ B, torch.tensor(0, dtype=torch.uint8, device=device)
    else:
        raise ValueError("Unsupported operation.")

def binary_to_decimal(binary_tensor):
    bit_string = "".join(map(str, binary_tensor.tolist()))
    return int(bit_string, 2)

##########################################
# Task Functions
##########################################
def matrix_multiply_task():
    A = torch.randn(64, 64, device=device)
    B = torch.randn(64, 64, device=device)
    start_time = time.perf_counter()
    result = torch.matmul(A, B)
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print("Matrix multiplication completed in {:.4f} seconds. (Mean of result: {:.4f})".format(
          elapsed, result.mean().item()))

def alu_addition_task():
    A = torch.randint(0, 2, (64,), dtype=torch.uint8, device=device)
    B = torch.randint(0, 2, (64,), dtype=torch.uint8, device=device)
    start_time = time.perf_counter()
    result, carry = alu_64bit_vectorized(A, B, "add")
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print("64-bit ALU addition completed in {:.4f} seconds.".format(elapsed))
    # Optionally, print binary and decimal results:
    print("Addition result (binary):", result)
    print("Addition result (decimal):", binary_to_decimal(result))
    print("Final Carry for addition:", carry)

##########################################
# Core Class: Represents a single processing core
##########################################
class Core:
    def __init__(self, core_id):
        self.core_id = core_id

    def execute_task(self, task_name, task_function):
        print(f"[Core {self.core_id}] Executing task: {task_name}")
        start_time = time.perf_counter()
        task_function()
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"[Core {self.core_id}] Task {task_name} completed in {elapsed * 1000:.2f} ms.")

##########################################
# Processor Class: Represents an 8-core processor with integrated chips
##########################################
class Processor:
    def __init__(self, num_cores=8):
        self.cores = [Core(i) for i in range(num_cores)]

    def run_tasks(self, tasks):
        # tasks: list of tuples (task_name, task_function)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.cores)) as executor:
            futures = []
            for i, (task_name, task_func) in enumerate(tasks):
                core = self.cores[i % len(self.cores)]
                futures.append(executor.submit(core.execute_task, task_name, task_func))
            for future in concurrent.futures.as_completed(futures):
                future.result()

##########################################
# Control Unit: Orchestrates tasks and inter-core scheduling
##########################################
class ControlUnit:
    def __init__(self):
        self.dependency_graph = {}  # Map task names to dependencies and functions
        self.completed_tasks = set()
    
    def add_task(self, task_name, dependencies, task_function):
        self.dependency_graph[task_name] = {'dependencies': dependencies, 'function': task_function}
    
    def task_ready(self, task_name):
        deps = self.dependency_graph[task_name]['dependencies']
        return all(dep in self.completed_tasks for dep in deps)
    
    def run(self, processor):
        # Simple scheduler: run tasks when ready, in dependency order.
        tasks_to_run = []
        while len(self.completed_tasks) < len(self.dependency_graph):
            for task_name, info in self.dependency_graph.items():
                if task_name not in self.completed_tasks and self.task_ready(task_name):
                    tasks_to_run.append((task_name, info['function']))
                    self.completed_tasks.add(task_name)
            if tasks_to_run:
                processor.run_tasks(tasks_to_run)
                tasks_to_run = []
            time.sleep(0.001)  # Small delay for scheduling

##########################################
# Main Execution: Build the full chip with 8 cores and integrated tasks
##########################################
if __name__ == "__main__":
    # Define tasks: Matrix multiplication first, then ALU addition (dependent on the first)
    tasks = [
        ("MatrixMultiplication", matrix_multiply_task),
        ("ALUAddition", alu_addition_task)
    ]
    
    # Initialize Control Unit and add tasks with dependencies
    control_unit = ControlUnit()
    control_unit.add_task("MatrixMultiplication", [], matrix_multiply_task)
    control_unit.add_task("ALUAddition", ["MatrixMultiplication"], alu_addition_task)
    
    # Initialize an 8-core Processor
    processor = Processor(num_cores=8)
    
    print("Starting full 8-core processor simulation with matrix-based computational engine...")
    control_unit.run(processor)
    print("Full processor simulation completed.")
