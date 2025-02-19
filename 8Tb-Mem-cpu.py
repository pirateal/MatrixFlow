import cupy as cp
import time
import os
import numpy as np
from tqdm import tqdm
import psutil
from unicorn import *
from unicorn.x86_const import *

##############################################
# Matrix Memory Setup
##############################################
chip_shape = (1024, 1024, 1024)  # Each chip ~1GB
num_chips = 8
total_capacity = chip_shape[0] * chip_shape[1] * chip_shape[2] * num_chips

# Create the simulated memory
ram_chips = cp.zeros((num_chips, *chip_shape), dtype=cp.uint8)

##############################################
# Memory Access Functions
##############################################
def write_block_to_chip(chip_id, start_index, data_block):
    chip_flat = ram_chips[chip_id].reshape(-1)
    chip_flat[start_index:start_index + data_block.size] = data_block

def read_block_from_chip(chip_id, start_index, block_size):
    chip_flat = ram_chips[chip_id].reshape(-1)
    return chip_flat[start_index:start_index + block_size]

##############################################
# File Storage Implementation
##############################################
def store_file_in_memory(file_path, chunk_bytes=1024*1024):
    file_size = os.path.getsize(file_path)
    print(f"Storing file: {file_path} ({file_size / 1e6:.2f} MB)")

    chip_capacity = chip_shape[0] * chip_shape[1] * chip_shape[2]
    if file_size > chip_capacity * num_chips:
        print("Warning: File size exceeds memory capacity!")
    
    with open(file_path, 'rb') as f:
        total_written = 0
        chip_id = 0
        chip_offset = 0
        start_time = time.time()

        with tqdm(total=file_size, unit='B', unit_scale=True) as pbar:
            while True:
                chunk = f.read(chunk_bytes)
                if not chunk:
                    break

                # Convert chunk to CuPy array directly as uint8
                chunk_np = np.frombuffer(chunk, dtype=np.uint8)
                chunk_cp = cp.array(chunk_np)

                bytes_remaining = chunk_cp.size
                pos = 0
                while bytes_remaining > 0:
                    if chip_id >= num_chips:
                        print("Memory is full!")
                        return total_written

                    available = chip_capacity - chip_offset
                    to_write = min(bytes_remaining, available)
                    
                    block = chunk_cp[pos:pos + to_write]
                    write_block_to_chip(chip_id, chip_offset, block)
                    
                    chip_offset += to_write
                    pos += to_write
                    bytes_remaining -= to_write
                    
                    if chip_offset >= chip_capacity:
                        chip_id += 1
                        chip_offset = 0

                total_written += chunk_cp.size
                pbar.update(len(chunk))
                
                if total_written % (10 * 1024 * 1024) < chunk_bytes:
                    elapsed = time.time() - start_time
                    throughput = total_written / elapsed / 1e6
                    cpu_usage = psutil.cpu_percent()
                    mem_info = psutil.virtual_memory()
                    gpu_mem = cp.get_default_memory_pool().used_bytes() / 1e6
                    print(f"\nStats: {throughput:.2f} MB/s, CPU: {cpu_usage:.1f}%, "
                          f"RAM: {mem_info.percent}%, GPU: {gpu_mem:.2f} MB")

                # Clear GPU memory periodically
                cp.get_default_memory_pool().free_all_blocks()

    total_time = time.time() - start_time
    print(f"\nCompleted: {total_time:.2f}s, Avg: {total_written/total_time/1e6:.2f} MB/s")
    return total_written

##############################################
# File Retrieval Implementation
##############################################
def retrieve_file_from_memory(output_path, total_bytes, chunk_size=64*1024*1024):
    print(f"Retrieving {total_bytes / 1e6:.2f} MB to {output_path}")
    
    with open(output_path, 'wb') as out_file:
        bytes_left = total_bytes
        chip_id = 0
        chip_offset = 0
        
        with tqdm(total=total_bytes, unit='B', unit_scale=True) as pbar:
            while bytes_left > 0 and chip_id < num_chips:
                chip_capacity = chip_shape[0] * chip_shape[1] * chip_shape[2]
                
                # Process data in chunks
                while chip_offset < chip_capacity and bytes_left > 0:
                    # Calculate how much to read in this iteration
                    available = min(chunk_size, chip_capacity - chip_offset)
                    to_read = min(bytes_left, available)
                    
                    # Read chunk and write directly
                    data = read_block_from_chip(chip_id, chip_offset, to_read)
                    out_file.write(cp.asnumpy(data).tobytes())
                    
                    bytes_left -= to_read
                    chip_offset += to_read
                    pbar.update(to_read)
                    
                    # Clear GPU memory after each chunk
                    del data
                    cp.get_default_memory_pool().free_all_blocks()
                
                if chip_offset >= chip_capacity:
                    chip_id += 1
                    chip_offset = 0
    
    print("Retrieval completed successfully")

##############################################
# Main Test Function with Emulation Integration
##############################################
def main():
    print("Matrix Memory System Test with Emulation Integration")
    print("=" * 50)
    
    # File storage test
    test_file = "The.Wizard.Of.Oz.1939.75th.Anniversary.Edition.1080p.BluRay.x264.anoXmous_.mp4"  # Change this to your input file path
    if not os.path.exists(test_file):
        print(f"Error: Input file {test_file} not found!")
        return
    
    # Store the file into memory
    total_bytes = store_file_in_memory(test_file)
    
    # Emulate reading the stored data as part of a computational process
    print("\nStarting emulation...")
    emu_total_capacity = total_capacity  # Limit emulation to memory capacity
    emu_start_addr = 0  # Starting address for emulation
    try:
        mu = Uc(UC_ARCH_X86, UC_MODE_64)
        mu.mem_map(emu_start_addr, emu_total_capacity)
        mu.reg_write(UC_X86_REG_RSP, 0x100000)  # Initialize stack pointer

        # Here you would emulate the process of accessing the memory
        print("Emulation started.")
        mu.emu_start(emu_start_addr, emu_start_addr + 0x1000)  # Example start
    except UcError as e:
        print(f"Emulation stopped: {e}")

    # Retrieve the file back from memory
    retrieve_file_from_memory("output.mp4", total_bytes)  # Change this to your desired output path

if __name__ == "__main__":
    main()
