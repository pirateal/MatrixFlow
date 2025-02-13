import torch
import time

# Set device: use CUDA if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MatrixMemorySubsystem:
    """
    Matrix-Based Memory Subsystem

    This component simulates a memory unit that includes:
      - A register file: a matrix of registers.
      - A direct-mapped cache: simulating fast-access cache memory.
      - Main memory: a larger memory space represented as a matrix.
    """

    def __init__(self, num_registers=16, register_width=64,
                 memory_size=256, cell_dtype=torch.int64,
                 num_cache_lines=8, block_size=4):
        # Register file: each register is a 64-bit integer (can be treated as a scalar register)
        self.register_file = torch.zeros(num_registers, dtype=cell_dtype, device=device)
        
        # Main memory: simulated as a 1D tensor of memory cells
        self.main_memory = torch.zeros(memory_size, dtype=cell_dtype, device=device)
        
        # Cache parameters
        self.num_cache_lines = num_cache_lines
        self.block_size = block_size

        # Cache data: each line holds a block of memory cells
        self.cache_data = torch.zeros((num_cache_lines, block_size), dtype=cell_dtype, device=device)
        # Cache tag: the starting address of the block stored in each cache line (-1 indicates invalid)
        self.cache_tags = -torch.ones(num_cache_lines, dtype=torch.int64, device=device)
        # Valid bits for each cache line (0: invalid, 1: valid)
        self.cache_valid = torch.zeros(num_cache_lines, dtype=torch.bool, device=device)

    def _cache_line_index(self, address):
        """Compute cache line index for a given memory address."""
        # For a direct-mapped cache, we use modulo on the block number
        block_num = address // self.block_size
        return block_num % self.num_cache_lines

    def _load_cache_line(self, address):
        """
        Load a block from main memory into the cache.
        Returns the cache line index.
        """
        line_index = self._cache_line_index(address)
        # Calculate the starting address of the block
        block_start = (address // self.block_size) * self.block_size
        # Clamp block_start in case we're near the end of main memory
        block_end = min(block_start + self.block_size, self.main_memory.shape[0])
        # Load block into cache (fill the rest with zeros if block is smaller than block_size)
        block = self.main_memory[block_start:block_end]
        if block.shape[0] < self.block_size:
            # Pad with zeros
            padded_block = torch.zeros(self.block_size, dtype=self.main_memory.dtype, device=device)
            padded_block[:block.shape[0]] = block
            block = padded_block
        self.cache_data[line_index] = block
        self.cache_tags[line_index] = block_start
        self.cache_valid[line_index] = True
        # Debug: print cache load event
        print(f"[CACHE] Loaded block starting at address {block_start} into cache line {line_index}.")
        return line_index

    def read_memory(self, address):
        """
        Read a value from main memory using the cache.
        If the value is in the cache (cache hit), return it directly.
        Otherwise, load the appropriate block into the cache (cache miss) and return the value.
        """
        if address < 0 or address >= self.main_memory.shape[0]:
            raise IndexError("Memory address out of range.")

        line_index = self._cache_line_index(address)
        # Check for cache hit: valid line and matching tag
        if self.cache_valid[line_index] and self.cache_tags[line_index] <= address < self.cache_tags[line_index] + self.block_size:
            offset = address - self.cache_tags[line_index]
            value = self.cache_data[line_index, offset].item()
            print(f"[MEM READ] Cache hit at address {address}: value = {value}")
            return value
        else:
            # Cache miss: load the block and then read
            self._load_cache_line(address)
            offset = address - self.cache_tags[line_index]
            value = self.cache_data[line_index, offset].item()
            print(f"[MEM READ] Cache miss at address {address}. Loaded block; value = {value}")
            return value

    def write_memory(self, address, value):
        """
        Write a value to main memory and update the cache if the corresponding block is present.
        """
        if address < 0 or address >= self.main_memory.shape[0]:
            raise IndexError("Memory address out of range.")

        # Write to main memory
        self.main_memory[address] = value
        print(f"[MEM WRITE] Written value {value} to main memory at address {address}.")

        # Update cache if the block is already loaded
        line_index = self._cache_line_index(address)
        if self.cache_valid[line_index] and self.cache_tags[line_index] <= address < self.cache_tags[line_index] + self.block_size:
            offset = address - self.cache_tags[line_index]
            self.cache_data[line_index, offset] = value
            print(f"[CACHE UPDATE] Updated cache line {line_index} at offset {offset} with value {value}.")

    def read_register(self, reg_index):
        """Read the value of a register from the register file."""
        if reg_index < 0 or reg_index >= self.register_file.shape[0]:
            raise IndexError("Register index out of range.")
        value = self.register_file[reg_index].item()
        print(f"[REG READ] Register {reg_index} = {value}")
        return value

    def write_register(self, reg_index, value):
        """Write a value to a register in the register file."""
        if reg_index < 0 or reg_index >= self.register_file.shape[0]:
            raise IndexError("Register index out of range.")
        self.register_file[reg_index] = value
        print(f"[REG WRITE] Written value {value} to register {reg_index}.")

    def transfer_data(self, src_address, dst_address, length):
        """
        Transfer a block of data from src_address to dst_address in main memory.
        This simulates a bus transfer operation using matrix slicing.
        """
        if (src_address < 0 or src_address + length > self.main_memory.shape[0] or
            dst_address < 0 or dst_address + length > self.main_memory.shape[0]):
            raise IndexError("Source or destination range is out of memory bounds.")

        # Read block from main memory
        data_block = self.main_memory[src_address:src_address+length].clone()
        # Write block to destination in main memory
        self.main_memory[dst_address:dst_address+length] = data_block
        print(f"[TRANSFER] Transferred data block of length {length} from address {src_address} to {dst_address}.")

    def debug_state(self):
        """Print the current state of the memory subsystem for debugging."""
        print("\n--- Matrix Memory Subsystem State ---")
        print("Register File:")
        print(self.register_file)
        print("\nMain Memory (first 32 cells):")
        print(self.main_memory[:32])
        print("\nCache State:")
        for i in range(self.num_cache_lines):
            valid = self.cache_valid[i].item()
            tag = self.cache_tags[i].item()
            print(f"Cache Line {i}: Valid = {valid}, Tag = {tag}, Data = {self.cache_data[i]}")
        print("---------------------------------------\n")


# === Test the Matrix-Based Memory Subsystem ===

def main_test():
    # Initialize the memory subsystem
    mem_subsystem = MatrixMemorySubsystem()

    # Write some values to main memory
    for addr in range(0, 16):
        mem_subsystem.write_memory(addr, addr * 10)

    # Read a few memory addresses to demonstrate cache misses/hits
    _ = mem_subsystem.read_memory(2)   # Likely a miss then hit within the block
    _ = mem_subsystem.read_memory(3)   # Should hit in the cache if within same block
    _ = mem_subsystem.read_memory(10)  # New block; miss then load

    # Test register file operations
    mem_subsystem.write_register(0, 12345)
    mem_subsystem.write_register(1, 67890)
    _ = mem_subsystem.read_register(0)
    _ = mem_subsystem.read_register(1)

    # Simulate a data transfer from one memory block to another
    mem_subsystem.transfer_data(src_address=0, dst_address=100, length=8)

    # Debug output: print the state of registers, main memory (partial), and cache
    mem_subsystem.debug_state()


if __name__ == '__main__':
    start_time = time.perf_counter()
    main_test()
    end_time = time.perf_counter()
    print(f"Matrix-Based Memory Subsystem Test Completed in {end_time - start_time:.4f} seconds.")
