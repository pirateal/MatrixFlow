import torch
import time

class MatrixCPU:
    def __init__(self, cores=16, matrix_size=1024):
        # Set up the number of cores and matrix size
        self.num_cores = cores
        self.matrix_size = matrix_size

        # Initialize GPU memory and set all values to zero
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory = torch.zeros((self.num_cores, self.matrix_size), dtype=torch.long, device=self.device)
        self.registers = torch.zeros((self.num_cores, 8), dtype=torch.long, device=self.device)  # 8 general registers
        self.flags = torch.zeros(self.num_cores, dtype=torch.long, device=self.device)  # Simulate flags (ZF, CF, etc.)
        self.stack = torch.zeros((self.num_cores, 1024), dtype=torch.long, device=self.device)  # Stack memory for function calls
        self.cycle_count = torch.zeros(self.num_cores, dtype=torch.long, device=self.device)  # Cycle counter for each core

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
        print(f"Core {core_id} completed program execution in {1000*(end_time - start_time)} ms. Total cycles: {self.cycle_count[core_id]}.")

    def _execute_instruction(self, core_id, instr):
        """Simulate execution of each instruction."""
        if instr[0] == 'LOAD':
            self.load(core_id, instr[1], instr[2])

        elif instr[0] == 'ADD':
            self.add(core_id, instr[1], instr[2], instr[3])

        elif instr[0] == 'MOV':
            self.mov(core_id, instr[1], instr[2])

        elif instr[0] == 'SUB':
            self.sub(core_id, instr[1], instr[2], instr[3])

        elif instr[0] == 'INC':
            self.inc(core_id, instr[1])

        elif instr[0] == 'DEC':
            self.dec(core_id, instr[1])

        elif instr[0] == 'CMP':
            self.cmp(core_id, instr[1], instr[2])

        elif instr[0] == 'JMP':
            self.jmp(core_id, instr[1])

        elif instr[0] == 'CALL':
            self.call(core_id, instr[1])

        elif instr[0] == 'RET':
            self.ret(core_id)

        elif instr[0] == 'HALT':
            print(f"Core {core_id} executed 'HALT' in 0 ns.")

        else:
            raise ValueError(f"Unknown instruction {instr[0]}")

    def load(self, core_id, addr, reg):
        """Simulate a memory load operation."""
        start_time = time.time()
        self.registers[core_id, reg] = self.memory[core_id, addr]
        self.cycle_count[core_id] += 1  # Simulate one cycle for load operation
        end_time = time.time()
        print(f"Core {core_id} executed 'LOAD' in {1000*(end_time - start_time)} ns.")

    def add(self, core_id, reg1, reg2, reg3):
        """Simulate an ADD operation using matrix-like addition."""
        start_time = time.time()
        self.registers[core_id, reg1] = self.registers[core_id, reg2] + self.registers[core_id, reg3]
        self.cycle_count[core_id] += 1  # Simulate one cycle for add operation
        end_time = time.time()
        print(f"Core {core_id} executed 'ADD' in {1000*(end_time - start_time)} ns.")

    def mov(self, core_id, reg1, reg2):
        """Simulate MOV operation (move data between registers)."""
        start_time = time.time()
        self.registers[core_id, reg1] = self.registers[core_id, reg2]
        self.cycle_count[core_id] += 1  # Simulate one cycle for move operation
        end_time = time.time()
        print(f"Core {core_id} executed 'MOV' in {1000*(end_time - start_time)} ns.")

    def sub(self, core_id, reg1, reg2, reg3):
        """Simulate a SUB operation (subtract registers)."""
        start_time = time.time()
        self.registers[core_id, reg1] = self.registers[core_id, reg2] - self.registers[core_id, reg3]
        self.cycle_count[core_id] += 1  # Simulate one cycle for subtract operation
        end_time = time.time()
        print(f"Core {core_id} executed 'SUB' in {1000*(end_time - start_time)} ns.")

    def inc(self, core_id, reg):
        """Simulate an INC operation (increment register)."""
        start_time = time.time()
        self.registers[core_id, reg] += 1
        self.cycle_count[core_id] += 1  # Simulate one cycle for increment operation
        end_time = time.time()
        print(f"Core {core_id} executed 'INC' in {1000*(end_time - start_time)} ns.")

    def dec(self, core_id, reg):
        """Simulate a DEC operation (decrement register)."""
        start_time = time.time()
        self.registers[core_id, reg] -= 1
        self.cycle_count[core_id] += 1  # Simulate one cycle for decrement operation
        end_time = time.time()
        print(f"Core {core_id} executed 'DEC' in {1000*(end_time - start_time)} ns.")

    def cmp(self, core_id, reg1, reg2):
        """Simulate CMP operation (compare two registers)."""
        start_time = time.time()
        result = self.registers[core_id, reg1] - self.registers[core_id, reg2]
        # Set flags: 0 = no zero, 1 = zero flag set
        self.flags[core_id] = 1 if result == 0 else 0  # Zero Flag (ZF)
        self.cycle_count[core_id] += 1  # Simulate one cycle for compare operation
        end_time = time.time()
        print(f"Core {core_id} executed 'CMP' in {1000*(end_time - start_time)} ns.")

    def jmp(self, core_id, address):
        """Simulate JMP operation (jump to a new address)."""
        print(f"Core {core_id} executed 'JMP' to address {address}.")
        self.cycle_count[core_id] += 1  # Simulate one cycle for jump operation

    def call(self, core_id, address):
        """Simulate CALL operation (function call)."""
        print(f"Core {core_id} executed 'CALL' to address {address}.")
        # Push return address to the stack
        self.stack[core_id, 0] = address  # Placeholder for return address (adjust as needed)
        self.cycle_count[core_id] += 1  # Simulate one cycle for call operation

    def ret(self, core_id):
        """Simulate RET operation (return from function)."""
        return_address = self.stack[core_id, 0]  # Pop return address from stack
        print(f"Core {core_id} executed 'RET' to address {return_address}.")
        self.cycle_count[core_id] += 1  # Simulate one cycle for return operation

# Example program with extended instructions
program = [
    ('MOV', 0, 1),  # Move register 1 to register 0
    ('ADD', 0, 1, 2),  # Add register 1 and 2, store in 0
    ('SUB', 1, 2, 3),  # Subtract register 2 and 3, store in 1
    ('CMP', 0, 1),  # Compare register 0 and 1
    ('INC', 2),     # Increment register 2
    ('DEC', 3),     # Decrement register 3
    ('JMP', 10),    # Jump to address 10
    ('CALL', 20),   # Call function at address 20
    ('RET',)        # Return from function
]

# Initialize the CPU
cpu = MatrixCPU()

# Run the program
cpu.run_program(program)
