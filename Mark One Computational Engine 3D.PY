import torch
import time
import numpy as np

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================
# Mark One Computational Engine (ALU)
# ==========================
class MarkOneALU:
    def __init__(self, size, bit_width=32):
        """
        Precompute lookup tables using an embedding-based approach with shared memory optimization.
        The lookup tables include arithmetic, bitwise, and comparison operations.
        """
        self.size = size
        values = torch.arange(size, device=device, dtype=torch.int32)
        self.tables = {
            'add': values.unsqueeze(0) + values.unsqueeze(1),
            'sub': values.unsqueeze(0) - values.unsqueeze(1),
            'mul': values.unsqueeze(0) * values.unsqueeze(1),
            'div': torch.where(values.unsqueeze(1) != 0,
                               values.unsqueeze(0).float() / values.unsqueeze(1).float(),
                               torch.tensor(float('inf'), device=device)),
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
        """
        Retrieve the result for the given operation using precomputed lookup tables.
        For the 'not' operation, b is not required.
        """
        if operation == 'not':
            return self.tables['not'][a]
        return self.tables[operation][a, b]

# ==========================
# 3D Particle Simulation Using Mark One ALU
# ==========================
def run_3d_simulation():
    # Parameters for the ALU and simulation
    size = 2048  # Range for lookup tables (adjustable as needed)
    alu = MarkOneALU(size)
    
    num_particles = 1024  # Number of particles in the simulation
    num_steps = 100       # Number of simulation time steps

    # Initialize particle positions and velocities for x, y, z as integers in [0, size)
    positions_x = torch.randint(0, size, (num_particles,), device=device, dtype=torch.int32)
    positions_y = torch.randint(0, size, (num_particles,), device=device, dtype=torch.int32)
    positions_z = torch.randint(0, size, (num_particles,), device=device, dtype=torch.int32)

    # Use relatively small velocities so that positions remain within range
    velocities_x = torch.randint(0, 50, (num_particles,), device=device, dtype=torch.int32)
    velocities_y = torch.randint(0, 50, (num_particles,), device=device, dtype=torch.int32)
    velocities_z = torch.randint(0, 50, (num_particles,), device=device, dtype=torch.int32)

    print("Starting 3D simulation using Mark One CPU for particle position updates...")

    start_sim_time = time.time()
    for step in range(num_steps):
        # Update positions: new_position = old_position + velocity using ALU 'add' operation.
        # Use modulo operation to keep positions within bounds.
        positions_x = alu.compute(positions_x, velocities_x, 'add') % size
        positions_y = alu.compute(positions_y, velocities_y, 'add') % size
        positions_z = alu.compute(positions_z, velocities_z, 'add') % size
    end_sim_time = time.time()

    print(f"3D simulation completed in {end_sim_time - start_sim_time:.6f} sec")
    print("Sample final positions (first 5 particles):")
    for i in range(5):
        print(f"Particle {i}: ( {positions_x[i].item()}, {positions_y[i].item()}, {positions_z[i].item()} )")

if __name__ == "__main__":
    run_3d_simulation()
