import torch
import time

# Ensure we are using the CUDA GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise Exception("This script requires a CUDA device.")

##########################################
# 1-bit Full Adder Function
##########################################
def full_adder_1bit(a, b, cin):
    """
    Computes the 1-bit full adder:
      sum_bit = a XOR b XOR cin
      cout = (a AND b) OR (cin AND (a XOR b))
    """
    sum_bit = a ^ b ^ cin
    cout = (a & b) | (cin & (a ^ b))
    return sum_bit, cout

##########################################
# 8-bit Full Adder Function
##########################################
def full_adder_8bit(A, B, Cin):
    """
    Processes 8 bits using a ripple-carry scheme.
    A, B: 8-element tensors (each element is 0 or 1, dtype=torch.uint8)
    Cin: a scalar carry-in as a tensor
    Returns an 8-bit sum tensor and the final carry as a tensor.
    """
    # Ensure A and B are contiguous and on the GPU
    A = A.to(device, dtype=torch.uint8).contiguous()
    B = B.to(device, dtype=torch.uint8).contiguous()
    
    # We'll build the sum bit-by-bit (LSB at index 7 and MSB at index 0)
    S = torch.empty(8, dtype=torch.uint8, device=device)
    carry = Cin.item()  # Get a Python scalar from carry tensor
    # Process bits from LSB to MSB (i.e. from index 7 down to 0)
    for j in reversed(range(8)):
        a_bit = A[j]
        b_bit = B[j]
        # Ensure the carry tensor is created properly on GPU
        c_tensor = torch.tensor(carry, dtype=torch.uint8, device=device).clone().detach()
        s_bit, carry = full_adder_1bit(a_bit, b_bit, c_tensor)
        S[j] = s_bit
    return S, torch.tensor(carry, dtype=torch.uint8, device=device).clone().detach()

##########################################
# 64-bit ALU Using Multiple 8-bit Adders
##########################################
def alu_64bit(A, B, operation="add"):
    """
    Processes 64-bit addition (or subtraction) by splitting the inputs
    into eight 8-bit chunks and chaining them (ripple-carry adder).
    A and B must be 64-element tensors of type torch.uint8.
    For addition and subtraction, the carry is propagated.
    For logical operations ("and", "or", "xor"), carry is ignored.
    Returns the 64-bit result and the final carry (for arithmetic).
    """
    if A.numel() != 64 or B.numel() != 64:
        raise ValueError("Both inputs must be 64 bits long.")

    # Ensure inputs are on GPU and contiguous
    A = A.to(device, dtype=torch.uint8, non_blocking=True).contiguous()
    B = B.to(device, dtype=torch.uint8, non_blocking=True).contiguous()
    
    # Split A and B into eight 8-bit chunks
    a_chunks = [A[i*8:(i+1)*8] for i in range(8)]
    b_chunks = [B[i*8:(i+1)*8] for i in range(8)]
    
    result_chunks = []
    carry = torch.tensor(0, dtype=torch.uint8, device=device)
    
    # For arithmetic operations, process from LSB to MSB (i.e. from chunk 7 to 0)
    # Here, we assume the LSB is the last chunk.
    for i in range(7, -1, -1):
        chunk_a = a_chunks[i]
        chunk_b = b_chunks[i]
        if operation == "add":
            sum_chunk, carry = full_adder_8bit(chunk_a, chunk_b, carry)
            result_chunks.insert(0, sum_chunk)  # Insert at beginning to maintain order
        elif operation == "subtract":
            # For subtraction, compute two's complement of B chunk and then add.
            # Two's complement for 8 bits: invert bits then add 1.
            b_inverted = 1 - chunk_b  # Inversion for binary (since bits are 0 and 1)
            # Add 1 to the inverted value using full adder with carry_in = 1
            sum_temp, carry_temp = full_adder_8bit(b_inverted, torch.zeros(8, dtype=torch.uint8, device=device), torch.tensor(1, dtype=torch.uint8, device=device))
            # Now perform addition: A + (two's complement of B)
            sum_chunk, carry = full_adder_8bit(chunk_a, sum_temp, carry)
            result_chunks.insert(0, sum_chunk)
        elif operation == "and":
            result_chunks.insert(0, chunk_a & chunk_b)
        elif operation == "or":
            result_chunks.insert(0, chunk_a | chunk_b)
        elif operation == "xor":
            result_chunks.insert(0, chunk_a ^ chunk_b)
        else:
            raise ValueError("Unsupported operation.")
    
    # Combine the result chunks to form the full 64-bit result
    result = torch.cat(result_chunks)
    return result, carry

##########################################
# Profiling and Testing the 64-bit ALU
##########################################
def binary_to_decimal(binary_tensor):
    # Convert a tensor of bits (e.g., [1, 0, 1, 1, ...]) to a decimal integer
    bit_string = "".join(map(str, binary_tensor.tolist()))
    return int(bit_string, 2)

def test_64bit_alu():
    # Create random 64-bit binary numbers (each tensor has 64 elements)
    A = torch.randint(0, 2, (64,), dtype=torch.uint8, device=device)
    B = torch.randint(0, 2, (64,), dtype=torch.uint8, device=device)
    
    # Profile the 64-bit addition using the GPU exclusively
    torch.cuda.synchronize()  # Ensure all GPU work is done
    start_time = time.perf_counter()
    result_add, carry_add = alu_64bit(A, B, "add")
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    exec_time_ns = (end_time - start_time) * 1e9  # in nanoseconds

    print("=== 64-bit ALU Test ===")
    print("Input A (64 bits):", A)
    print("Input B (64 bits):", B)
    print("Addition result (binary):", result_add)
    print("Addition result (decimal):", binary_to_decimal(result_add))
    print("Final Carry for addition:", carry_add)
    print(f"Execution time for 64-bit addition: {exec_time_ns:.2f} ns")
    
    # Similarly, you can test subtraction, AND, OR, XOR as needed.

if __name__ == "__main__":
    test_64bit_alu()
