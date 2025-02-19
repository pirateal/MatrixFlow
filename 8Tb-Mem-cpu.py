import cupy as cp
import time
import os
import numpy as np
from tqdm import tqdm
import psutil
from unicorn import *
from unicorn.x86_const import *

##############################################
# Matrix Memory Setup (Simulated 8-chip memory)
##############################################
# For simulation purposes, each chip is a 3D array of shape (1024, 1024, 1024) = ~1GB.
# (In a real design, these numbers would be scaled up for 1TB per chip.)
chip_shape = (1024, 1024, 1024)  # Each chip ~1GB (1024^3 bytes)
num_chips = 8
total_capacity = chip_shape[0] * chip_shape[1] * chip_shape[2] * num_chips  # Total capacity in bytes

# Create the simulated memory as a CuPy array for each chip.
# Using uint8 to simulate FP8 values.
ram_chips = cp.zeros((num_chips, *chip_shape), dtype=cp.uint8)

##############################################
# Helper Functions for Matrix Memory Access
##############################################
def write_block_to_chip(chip_id, start_index, data_block):
    """
    Write a block of data (1D cp.array of dtype uint8) to a flattened view
    of the chip's memory starting at start_index.
    """
    chip_flat = ram_chips[chip_id].reshape(-1)
    chip_flat[start_index:start_index + data_block.size] = data_block

def read_block_from_chip(chip_id, start_index, block_size):
    chip_flat = ram_chips[chip_id].reshape(-1)
    return chip_flat[start_index:start_index + block_size]

##############################################
# Simulated FP8 Matrix Logic Operation
##############################################
def simulate_fp8_conversion(data_block):
    """
    Simulate an FP8 conversion on the data block.
    Here we perform a dummy operation to mimic computation overhead.
    """
    d = cp.asarray(data_block, dtype=cp.float32)
    # Dummy operation: for instance, scaling (this is where real FP8 quantization could occur)
    d = d * 1.0  
    result = cp.clip(cp.rint(d), 0, 255).astype(cp.uint8)
    return result

##############################################
# Optimized File Storage with Progress & Stats
##############################################
def store_file_in_memory(file_path, chunk_bytes=1024*1024):
    """
    Reads a file in chunks, simulates FP8 conversion on each chunk,
    and writes the data to simulated matrix memory in a vectorized fashion.
    Displays progress, throughput, and resource statistics.
    """
    file_size = os.path.getsize(file_path)
    print(f"Storing file: {file_path} ({file_size / 1e6:.2f} MB) into simulated memory...")

    chip_capacity = chip_shape[0] * chip_shape[1] * chip_shape[2]
    total_capacity_local = chip_capacity * num_chips
    if file_size > total_capacity_local:
        print("Warning: File size exceeds simulated memory capacity!")
    
    with open(file_path, 'rb') as f:
        total_written = 0
        chip_id = 0
        chip_offset = 0  # offset within current chip (flattened index)
        start_time = time.time()

        with tqdm(total=file_size, unit='B', unit_scale=True) as pbar:
            while True:
                chunk = f.read(chunk_bytes)
                if not chunk:
                    break

                # Convert chunk using NumPy, then transfer to CuPy
                chunk_np = np.frombuffer(chunk, dtype=np.uint8)
                chunk_cp = cp.array(chunk_np)
                converted = simulate_fp8_conversion(chunk_cp)

                bytes_remaining = converted.size
                pos = 0
                while bytes_remaining > 0:
                    available = chip_capacity - chip_offset
                    if available <= 0:
                        chip_id += 1
                        chip_offset = 0
                        available = chip_capacity
                        if chip_id >= num_chips:
                            print("Simulated memory is full!")
                            break

                    to_write = min(bytes_remaining, available)
                    block = converted[pos: pos + to_write]
                    write_block_to_chip(chip_id, chip_offset, block)
                    chip_offset += to_write
                    pos += to_write
                    bytes_remaining -= to_write

                total_written += converted.size
                pbar.update(converted.size)
                
                # Display throughput and resource usage every 10 MB processed
                if total_written % (10 * 1024 * 1024) < chunk_bytes:
                    elapsed = time.time() - start_time
                    throughput = total_written / elapsed / 1e6  # MB/s
                    cpu_usage = psutil.cpu_percent(interval=0.1)
                    mem_info = psutil.virtual_memory()
                    gpu_mem_used = cp.get_default_memory_pool().used_bytes() / 1e6  # MB
                    print(f"Processed {total_written/1e6:.2f} MB, Throughput: {throughput:.2f} MB/s, "
                          f"CPU: {cpu_usage:.1f}% Memory: {mem_info.percent}% GPU Mem: {gpu_mem_used:.2f} MB")
                    
                if chip_id >= num_chips:
                    break

    total_time = time.time() - start_time
    avg_throughput = total_written / total_time / 1e6
    print(f"Completed storing file. Total time: {total_time:.2f} s, Average Throughput: {avg_throughput:.2f} MB/s")
    return total_written

##############################################
# File Retrieval
##############################################
def retrieve_file_from_memory(output_path, total_bytes):
    """
    Reads back data from simulated memory sequentially until total_bytes are retrieved,
    and writes it to the output file.
    """
    print(f"Retrieving file ({total_bytes / 1e6:.2f} MB) from simulated memory...")
    with open(output_path, 'wb') as out_file:
        bytes_left = total_bytes
        chip_id = 0
        chip_offset = 0
        while bytes_left > 0 and chip_id < num_chips:
            available = (chip_shape[0] * chip_shape[1] * chip_shape[2]) - chip_offset
            to_read = min(bytes_left, available)
            block = cp.asnumpy(read_block_from_chip(chip_id, chip_offset, to_read))
            out_file.write(block.tobytes())
            bytes_left -= to_read
            chip_offset += to_read
            if chip_offset >= chip_shape[0] * chip_shape[1] * chip_shape[2]:
                chip_id += 1
                chip_offset = 0
    print("File retrieval completed.")

##############################################
# Unicorn CPU Simulation (Minimal for Demonstration)
##############################################
def run_x86_program(uc, memory):
    # Map program code at 0x1000 (outside custom memory region)
    uc.mem_map(0x1000, 0x1000)
    uc.mem_write(0x1000, memory)
    
    uc.reg_write(UC_X86_REG_EAX, 0)
    uc.reg_write(UC_X86_REG_EBX, 5)
    uc.reg_write(UC_X86_REG_ECX, 10)
    
    uc.emu_start(0x1000, 0x1000 + len(memory))
    result_eax = uc.reg_read(UC_X86_REG_EAX)
    print(f"Result of ADD operation: EAX = {result_eax}")

##############################################
# Custom Memory Mapping for Unicorn
##############################################
def custom_mem_map(uc, start_address, memory_size):
    uc.mem_map(start_address, memory_size)

    def hook_mem_read(uc, access, address, size, value, user_data):
        if start_address <= address < start_address + memory_size:
            offset = address - start_address
            chip_id = offset // (chip_shape[0] * chip_shape[1] * chip_shape[2])
            local_offset = offset % (chip_shape[0] * chip_shape[1] * chip_shape[2])
            x = local_offset // (chip_shape[1] * chip_shape[2])
            y = (local_offset // chip_shape[2]) % chip_shape[1]
            z = local_offset % chip_shape[2]
            ret_val = int(read_block_from_chip(chip_id, x * chip_shape[1] * chip_shape[2] + y * chip_shape[2] + z, 1)[0])
            return ret_val
        return None

    def hook_mem_write(uc, access, address, size, value, user_data):
        if start_address <= address < start_address + memory_size:
            offset = address - start_address
            chip_id = offset // (chip_shape[0] * chip_shape[1] * chip_shape[2])
            local_offset = offset % (chip_shape[0] * chip_shape[1] * chip_shape[2])
            x = local_offset // (chip_shape[1] * chip_shape[2])
            y = (local_offset // chip_shape[2]) % chip_shape[1]
            z = local_offset % chip_shape[2]
            write_block_to_chip(chip_id, x * chip_shape[1] * chip_shape[2] + y * chip_shape[2] + z, cp.array([value], dtype=cp.uint8))
        return True

    uc.hook_add(UC_HOOK_MEM_READ, hook_mem_read)
    uc.hook_add(UC_HOOK_MEM_WRITE, hook_mem_write)

##############################################
# Main Stress Test Function
##############################################
def stress_test():
    # Initialize Unicorn emulator (32-bit mode)
    uc = Uc(UC_ARCH_X86, UC_MODE_32)
    x86_code = b"\xB8\x05\x00\x00\x00\xBB\x0A\x00\x00\x00\x01\xD8"  # Simple ADD operation

    print("Starting stress test with file I/O and matrix logic FP8...")

    # Map custom memory at 0x10000000 (for Unicorn to access, if needed)
    custom_mem_map(uc, 0x10000000, total_capacity)

    # Specify the large file to be processed (update path as needed)
    file_path = 'The.Wizard.Of.Oz.1939.75th.Anniversary.Edition.1080p.BluRay.x264.anoXmous_.mp4'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Store file in simulated memory
    total_bytes = store_file_in_memory(file_path)

    # Run a basic x86 program (for demonstration)
    run_x86_program(uc, x86_code)

    # Retrieve the file from simulated memory
    retrieve_file_from_memory('retrieved_file.mp4', total_bytes)

# Run the stress test and file operations
stress_test()
