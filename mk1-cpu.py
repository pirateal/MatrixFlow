import torch
import time
import numpy as np

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define matrix-based lookup for arithmetic and bitwise operations using shared memory
class MarkOneALU:
    def __init__(self, size, bit_width=32):
        """Precompute lookup tables using an embedding-based approach with shared memory optimization."""
        self.size = size
        values = torch.arange(size, device='cuda', dtype=torch.int32)
        
        self.tables = {
            'add': values.unsqueeze(0) + values.unsqueeze(1),
            'sub': values.unsqueeze(0) - values.unsqueeze(1),
            'mul': values.unsqueeze(0) * values.unsqueeze(1),
            'div': torch.where(values.unsqueeze(1) != 0, values.unsqueeze(0).float() / values.unsqueeze(1).float(), torch.tensor(float('inf'), device='cuda')),
            'and': values.unsqueeze(0) & values.unsqueeze(1),
            'or': values.unsqueeze(0) | values.unsqueeze(1),
            'xor': values.unsqueeze(0) ^ values.unsqueeze(1),
            'lshift': values.unsqueeze(0) << (values.unsqueeze(1) & (bit_width - 1)),
            'rshift': values.unsqueeze(0) >> (values.unsqueeze(1) & (bit_width - 1)),
            'not': ~values,
            'gt': (values.unsqueeze(0) > values.unsqueeze(1)).to(torch.int32),
            'lt': (values.unsqueeze(0) < values.unsqueeze(1)).to(torch.int32),
            'eq': (values.unsqueeze(0) == values.unsqueeze(1)).to(torch.int32)
        }

    def compute(self, a, b, operation):
        """Performs ALU operations using precomputed lookup tables with shared memory."""
        if operation == 'not':
            return self.tables['not'][a]
        return self.tables[operation][a, b]

# Initialize shared memory ALU
size = 1024
alu = MarkOneALU(size)

# Define batch sizes to test
batch_sizes = [256, 512, 1024, 2048, 4096]
results_collection = {}

for batch_size in batch_sizes:
    a_batch = torch.randint(0, size, (batch_size,), device='cuda', dtype=torch.int32)
    b_batch = torch.randint(0, size, (batch_size,), device='cuda', dtype=torch.int32)
    
    start_time = time.time()
    results = {
        'add': alu.compute(a_batch, b_batch, 'add'),
        'sub': alu.compute(a_batch, b_batch, 'sub').abs(),
        'mul': alu.compute(a_batch, b_batch, 'mul'),
        'div': alu.compute(a_batch, b_batch, 'div'),
        'and': alu.compute(a_batch, b_batch, 'and'),
        'or': alu.compute(a_batch, b_batch, 'or'),
        'xor': alu.compute(a_batch, b_batch, 'xor'),
        'lshift': alu.compute(a_batch, b_batch, 'lshift'),
        'rshift': alu.compute(a_batch, b_batch, 'rshift'),
        'not': alu.compute(a_batch, None, 'not'),
        'gt': alu.compute(a_batch, b_batch, 'gt'),
        'lt': alu.compute(a_batch, b_batch, 'lt'),
        'eq': alu.compute(a_batch, b_batch, 'eq')
    }
    end_time = time.time()
    
    results_collection[batch_size] = end_time - start_time
    
    # Output results and computation time for each batch
    print(f"Batch Size: {batch_size}, ALU Computation Time: {end_time - start_time:.6f} sec")
    print("Sample Results:")
    for i in range(3):  # Display first 3 results per batch size
        print(f"{a_batch[i].item()} + {b_batch[i].item()} = {results['add'][i].item()}")
        print(f"{a_batch[i].item()} - {b_batch[i].item()} = {results['sub'][i].item()}")
        print(f"{a_batch[i].item()} * {b_batch[i].item()} = {results['mul'][i].item()}")
        print(f"{a_batch[i].item()} / {b_batch[i].item()} = {results['div'][i].item()}")
        print(f"{a_batch[i].item()} & {b_batch[i].item()} = {results['and'][i].item()}")
        print(f"{a_batch[i].item()} | {b_batch[i].item()} = {results['or'][i].item()}")
        print(f"{a_batch[i].item()} ^ {b_batch[i].item()} = {results['xor'][i].item()}")
        print(f"{a_batch[i].item()} << {b_batch[i].item()} = {results['lshift'][i].item()}")
        print(f"{a_batch[i].item()} >> {b_batch[i].item()} = {results['rshift'][i].item()}")
        print(f"~{a_batch[i].item()} = {results['not'][i].item()}")
        print(f"{a_batch[i].item()} > {b_batch[i].item()} = {results['gt'][i].item()}")
        print(f"{a_batch[i].item()} < {b_batch[i].item()} = {results['lt'][i].item()}")
        print(f"{a_batch[i].item()} == {b_batch[i].item()} = {results['eq'][i].item()}")
        print("-")

# Display the best batch size based on performance
best_batch = min(results_collection, key=results_collection.get)
print(f"Best Batch Size: {best_batch} with Computation Time: {results_collection[best_batch]:.6f} sec")
