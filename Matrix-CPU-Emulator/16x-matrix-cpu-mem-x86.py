import torch
import time

class MatrixCPU:
    def __init__(self, cores=16, matrix_size=1024):
        # Set up the number of cores and matrix size
        self.num_cores = cores
        self.matrix_size = matrix_size

        # Initialize GPU memory and set all values to zero
        self.memory = torch.zeros((self.num_cores, self.matrix_size), dtype=torch.long, device='cuda')
        self.registers = torch.zeros((self.num_cores, self.matrix_size), dtype=torch.long, device='cuda')

    def run_program(self, program):
        """Simulate running a program on the CPU."""
        for core_id in range(self.num_cores):
            self.execute(core_id, program)

    def execute(self, core_id, program):
        """Execute a sequence of instructions."""
        start_time = time.time()

        for instr in program:
            self._execute_instruction(core_id, instr)

        end_time = time.time()
        print(f"Core {core_id} completed program execution in {1000*(end_time - start_time)} ms.")

    def _execute_instruction(self, core_id, instr):
        """Simulate execution of each instruction."""
        if instr[0] == 'LOAD':
            self.load(core_id, instr[1], instr[2])

        elif instr[0] == 'ADD':
            self.add(core_id, instr[1], instr[2], instr[3])

        elif instr[0] == 'MUL':
            self.mul(core_id, instr[1], instr[2], instr[3])

        elif instr[0] == 'SUB':
            self.sub(core_id, instr[1], instr[2], instr[3])

        elif instr[0] == 'STORE':
            self.store(core_id, instr[1], instr[2])

        elif instr[0] == 'HALT':
            print(f"Core {core_id} executed 'HALT' in 0 ns.")

        else:
            raise ValueError(f"Unknown instruction {instr[0]}")

    def load(self, core_id, addr, reg):
        """Simulate a memory load operation."""
        start_time = time.time()
        self.registers[core_id, reg] = self.memory[core_id, addr]
        end_time = time.time()
        print(f"Core {core_id} executed 'LOAD' in {1000*(end_time - start_time)} ns.")

    def add(self, core_id, reg1, reg2, reg3):
        """Simulate an ADD operation."""
        start_time = time.time()
        self.registers[core_id, reg1] = self.registers[core_id, reg2] + self.registers[core_id, reg3]
        end_time = time.time()
        print(f"Core {core_id} executed 'ADD' in {1000*(end_time - start_time)} ns.")

    def mul(self, core_id, reg1, reg2, reg3):
        """Simulate a MUL operation."""
        start_time = time.time()
        self.registers[core_id, reg1] = self.registers[core_id, reg2] * self.registers[core_id, reg3]
        end_time = time.time()
        print(f"Core {core_id} executed 'MUL' in {1000*(end_time - start_time)} ns.")

    def sub(self, core_id, reg1, reg2, reg3):
        """Simulate a SUB operation."""
        start_time = time.time()
        self.registers[core_id, reg1] = self.registers[core_id, reg2] - self.registers[core_id, reg3]
        end_time = time.time()
        print(f"Core {core_id} executed 'SUB' in {1000*(end_time - start_time)} ns.")

    def store(self, core_id, addr, reg):
        """Simulate storing a value to memory."""
        start_time = time.time()
        self.memory[core_id, addr] = self.registers[core_id, reg]
        end_time = time.time()
        print(f"Core {core_id} executed 'STORE' in {1000*(end_time - start_time)} ns.")

# Example program with 16 cores
program = [
    ('LOAD', 0, 0),  # Load from memory[core_id, 0] into register 0
    ('ADD', 1, 0, 0),  # Add register 0 and 0, store result in register 1
    ('MUL', 2, 0, 1),  # Multiply register 0 and 1, store result in register 2
    ('SUB', 3, 1, 2),  # Subtract register 2 from register 1, store result in register 3
    ('STORE', 0, 3),  # Store the result from register 3 to memory at address 0
    ('HALT',)  # Halt the execution
]

# Initialize the matrix CPU with 16 cores
matrix_cpu = MatrixCPU(cores=16, matrix_size=1024)

# Run the program
matrix_cpu.run_program(program)
