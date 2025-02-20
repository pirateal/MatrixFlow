import torch
import time

# Set device: use CUDA if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Memory Subsystem Component
# =========================
class MemorySubsystem:
    """
    Matrix-Based Memory Subsystem

    Simulates a memory unit including:
      - A register file (simple 1D tensor of registers)
      - Main memory (a 1D tensor of memory cells)
      - A simple direct-mapped cache for faster memory accesses
    """
    def __init__(self, num_registers=8, memory_size=128, cell_dtype=torch.int64,
                 num_cache_lines=4, block_size=4):
        # Register file: one register per element
        self.register_file = torch.zeros(num_registers, dtype=cell_dtype, device=device)
        # Main memory: a larger 1D tensor for data storage
        self.main_memory = torch.zeros(memory_size, dtype=cell_dtype, device=device)
        # Cache parameters
        self.num_cache_lines = num_cache_lines
        self.block_size = block_size
        # Cache storage and metadata
        self.cache_data = torch.zeros((num_cache_lines, block_size), dtype=cell_dtype, device=device)
        self.cache_tags = -torch.ones(num_cache_lines, dtype=torch.int64, device=device)
        self.cache_valid = torch.zeros(num_cache_lines, dtype=torch.bool, device=device)

    def _cache_line_index(self, address):
        block_num = address // self.block_size
        return block_num % self.num_cache_lines

    def _load_cache_line(self, address):
        line_index = self._cache_line_index(address)
        block_start = (address // self.block_size) * self.block_size
        block_end = min(block_start + self.block_size, self.main_memory.shape[0])
        block = self.main_memory[block_start:block_end]
        if block.shape[0] < self.block_size:
            padded_block = torch.zeros(self.block_size, dtype=self.main_memory.dtype, device=device)
            padded_block[:block.shape[0]] = block
            block = padded_block
        self.cache_data[line_index] = block
        self.cache_tags[line_index] = block_start
        self.cache_valid[line_index] = True
        # Debug info: indicate cache load
        print(f"[CACHE] Loaded block starting at address {block_start} into cache line {line_index}.")
        return line_index

    def read_memory(self, address):
        if address < 0 or address >= self.main_memory.shape[0]:
            raise IndexError("Memory address out of range.")
        line_index = self._cache_line_index(address)
        # Cache hit: valid line and address within block
        if self.cache_valid[line_index] and self.cache_tags[line_index] <= address < self.cache_tags[line_index] + self.block_size:
            offset = address - self.cache_tags[line_index]
            value = self.cache_data[line_index, offset].item()
            print(f"[MEM READ] Cache hit at address {address}: value = {value}")
            return value
        else:
            # Cache miss: load block and then read
            self._load_cache_line(address)
            offset = address - self.cache_tags[line_index]
            value = self.cache_data[line_index, offset].item()
            print(f"[MEM READ] Cache miss at address {address}. Loaded block; value = {value}")
            return value

    def write_memory(self, address, value):
        if address < 0 or address >= self.main_memory.shape[0]:
            raise IndexError("Memory address out of range.")
        self.main_memory[address] = value
        print(f"[MEM WRITE] Written value {value} to main memory at address {address}.")
        line_index = self._cache_line_index(address)
        if self.cache_valid[line_index] and self.cache_tags[line_index] <= address < self.cache_tags[line_index] + self.block_size:
            offset = address - self.cache_tags[line_index]
            self.cache_data[line_index, offset] = value
            print(f"[CACHE UPDATE] Updated cache line {line_index} at offset {offset} with value {value}.")

    def read_register(self, reg_index):
        if reg_index < 0 or reg_index >= self.register_file.shape[0]:
            raise IndexError("Register index out of range.")
        value = self.register_file[reg_index].item()
        print(f"[REG READ] Register {reg_index} = {value}")
        return value

    def write_register(self, reg_index, value):
        if reg_index < 0 or reg_index >= self.register_file.shape[0]:
            raise IndexError("Register index out of range.")
        self.register_file[reg_index] = value
        print(f"[REG WRITE] Written value {value} to register {reg_index}.")

# =========================
# ALU Component
# =========================
class ALU:
    """
    Matrix-Based ALU

    Provides basic arithmetic operations using PyTorch's GPU-accelerated operations.
    """
    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b

    def mul(self, a, b):
        return a * b

    def matmul(self, A, B):
        return torch.matmul(A, B)

    def transpose(self, A):
        return A.t()

# =========================
# Control Unit Component
# =========================
class ControlUnit:
    """
    Control Unit

    Manages fetching, decoding, and executing a simple instruction set.
    """
    def __init__(self, memory: MemorySubsystem, alu: ALU):
        self.memory = memory
        self.alu = alu
        self.program_counter = 0
        self.halted = False
        self.program = []  # Program is a list of instructions

    def load_program(self, program):
        self.program = program
        self.program_counter = 0
        self.halted = False

    def fetch_instruction(self):
        if self.program_counter < len(self.program):
            instr = self.program[self.program_counter]
            self.program_counter += 1
            return instr
        else:
            self.halted = True
            return None

    def execute_instruction(self, instr):
        opcode = instr[0]
        if opcode == "LOAD":
            # Format: ("LOAD", reg, mem_addr)
            _, reg, addr = instr
            value = self.memory.read_memory(addr)
            self.memory.write_register(reg, value)
            print(f"[INSTR] LOAD: Register {reg} <= Memory[{addr}] ({value})")
        elif opcode == "STORE":
            # Format: ("STORE", reg, mem_addr)
            _, reg, addr = instr
            value = self.memory.read_register(reg)
            self.memory.write_memory(addr, value)
            print(f"[INSTR] STORE: Memory[{addr}] <= Register {reg} ({value})")
        elif opcode == "ADD":
            # Format: ("ADD", dest_reg, src_reg1, src_reg2)
            _, dest, src1, src2 = instr
            a = self.memory.read_register(src1)
            b = self.memory.read_register(src2)
            result = self.alu.add(a, b)
            self.memory.write_register(dest, result)
            print(f"[INSTR] ADD: Register {dest} <= {a} + {b} = {result}")
        elif opcode == "SUB":
            # Format: ("SUB", dest_reg, src_reg1, src_reg2)
            _, dest, src1, src2 = instr
            a = self.memory.read_register(src1)
            b = self.memory.read_register(src2)
            result = self.alu.sub(a, b)
            self.memory.write_register(dest, result)
            print(f"[INSTR] SUB: Register {dest} <= {a} - {b} = {result}")
        elif opcode == "MUL":
            # Format: ("MUL", dest_reg, src_reg1, src_reg2)
            _, dest, src1, src2 = instr
            a = self.memory.read_register(src1)
            b = self.memory.read_register(src2)
            result = self.alu.mul(a, b)
            self.memory.write_register(dest, result)
            print(f"[INSTR] MUL: Register {dest} <= {a} * {b} = {result}")
        elif opcode == "HALT":
            self.halted = True
            print("[INSTR] HALT: Halting execution.")
        else:
            print(f"[INSTR] Unknown instruction: {opcode}")

    def run(self):
        print("[CONTROL] Starting program execution.")
        while not self.halted:
            instr = self.fetch_instruction()
            if instr is None:
                break
            self.execute_instruction(instr)
        print("[CONTROL] Program execution finished.")

# =========================
# Processing Unit: Merging Everything
# =========================
class ProcessingUnit:
    """
    Processing Unit

    Integrates the MemorySubsystem, ALU, and ControlUnit into a single simulated CPU.
    """
    def __init__(self):
        self.memory = MemorySubsystem()
        self.alu = ALU()
        self.control_unit = ControlUnit(self.memory, self.alu)

    def load_program(self, program):
        self.control_unit.load_program(program)

    def run(self):
        self.control_unit.run()
        self.debug_state()

    def debug_state(self):
        print("\n--- Processing Unit State ---")
        print("Registers:")
        print(self.memory.register_file)
        print("Main Memory (first 16 cells):")
        print(self.memory.main_memory[:16])
        print("-----------------------------\n")

# =========================
# Main Test: Running a Sample Program
# =========================
def main():
    # Create a processing unit
    pu = ProcessingUnit()

    # Initialize main memory with some demo values
    for addr in range(8):
        pu.memory.write_memory(addr, addr * 5)

    # Define a simple program (a list of instructions)
    # Instruction format examples:
    #   ("LOAD", dest_reg, mem_addr)  -> Load value from memory into register
    #   ("STORE", src_reg, mem_addr)  -> Store register value into memory
    #   ("ADD", dest_reg, src_reg1, src_reg2) -> Add two registers
    #   ("SUB", dest_reg, src_reg1, src_reg2) -> Subtract two registers
    #   ("MUL", dest_reg, src_reg1, src_reg2) -> Multiply two registers
    #   ("HALT",) -> Stop execution
    program = [
        ("LOAD", 0, 0),     # Load memory[0] into register 0
        ("LOAD", 1, 1),     # Load memory[1] into register 1
        ("ADD", 2, 0, 1),   # Add registers 0 and 1, store result in register 2
        ("MUL", 3, 0, 1),   # Multiply registers 0 and 1, store result in register 3
        ("SUB", 4, 3, 2),   # Subtract register 2 from register 3, store in register 4
        ("STORE", 4, 10),   # Store register 4 into memory[10]
        ("HALT",)           # Halt the program
    ]

    # Load and run the program on the processing unit
    pu.load_program(program)
    start_time = time.perf_counter()
    pu.run()
    end_time = time.perf_counter()
    print(f"Processing Unit Execution Time: {end_time - start_time:.4f} seconds.")

if __name__ == '__main__':
    main()
