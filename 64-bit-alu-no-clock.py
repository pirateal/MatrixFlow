import torch
import time

# Use CUDA if available, else fallback to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.int64  # Use int64 on CUDA

class ORXORALU:
    def __init__(self, batch_size=1024):
        self.batch_size = batch_size
        # Create a mask with all 64 bits set.
        # For torch.int64, ~0 (i.e. bitwise NOT of 0) gives -1, which in two's complement
        # is equivalent to a 64-bit value with all bits set.
        self.mask = ~torch.tensor(0, dtype=dtype, device=device)
        # Initialize A and B with values that fit within the 63-bit positive range,
        # avoiding negative numbers from the start.
        self.A = torch.randint(0, 9223372036854775807, (batch_size,), dtype=dtype, device=device)
        self.B = torch.randint(1, 9223372036854775807, (batch_size,), dtype=dtype, device=device)  # Avoid zero if needed
    
    def fast_addition(self, A, B, Cin=None):
        """Perform addition using OR and XOR gates with 64-bit masks."""
        if Cin is None:
            Cin = torch.zeros_like(A, dtype=dtype, device=device)
        Sum = (A ^ B ^ Cin) & self.mask
        Carry = ((A & B) | (Cin & (A ^ B))) & self.mask
        return Sum, Carry
    
    def fast_subtraction(self, A, B):
        """Perform subtraction using two's complement with 64-bit mask."""
        B_complement = ((~B + 1)) & self.mask
        return self.fast_addition(A, B_complement, Cin=torch.zeros_like(A, dtype=dtype, device=device))
    
    def bitwise_and_approx(self, A, B):
        """Approximate AND using OR and XOR with 64-bit mask."""
        return ((A | B) ^ (A ^ B)) & self.mask
    
    def left_shift(self, A, shift_by=1):
        """Left Shift with 64-bit overflow handling."""
        return (A << shift_by) & self.mask
    
    def right_shift(self, A, shift_by=1):
        """Right Shift with 64-bit mask."""
        return (A >> shift_by) & self.mask
    
    def multiplexer(self, A, B, select):
        """Multiplexer with 64-bit mask."""
        return ((A & (~select)) | (B & select)) & self.mask
    
    def bitwise_nor(self, A, B):
        """NOR with 64-bit mask."""
        return (~(A | B)) & self.mask
    
    def bitwise_xnor(self, A, B):
        """XNOR with 64-bit mask."""
        return (~(A ^ B)) & self.mask
    
    def execute(self):
        """Execute all ALU operations."""
        Sum, Carry = self.fast_addition(self.A, self.B)
        Sub, _ = self.fast_subtraction(self.A, self.B)
        AND_Approx = self.bitwise_and_approx(self.A, self.B)
        Left_Shifted = self.left_shift(self.A)
        Right_Shifted = self.right_shift(self.A)
        MUX_Output = self.multiplexer(self.A, self.B, Carry)
        NOR_Result = self.bitwise_nor(self.A, self.B)
        XNOR_Result = self.bitwise_xnor(self.A, self.B)
        
        return (Sum, Carry, Sub, AND_Approx, Left_Shifted, Right_Shifted, MUX_Output, NOR_Result, XNOR_Result)
    
    def benchmark(self):
        """Benchmark ALU performance."""
        start = time.time()
        for _ in range(1000):
            self.execute()
        end = time.time()
        print(f"OR/XOR ALU Performance: {end - start:.6f} seconds for 1000 executions")

# Instantiate and benchmark the ALU
alu = ORXORALU(batch_size=1024)
alu.benchmark()

# Sample Output
print("Sample Output:")
labels = ["Sum", "Carry", "Subtraction", "Approx AND", "Left Shift",
          "Right Shift", "MUX Output", "NOR", "XNOR"]
results = alu.execute()
for label, result in zip(labels, results):
    print(f"{label}:", result[:10].tolist())
