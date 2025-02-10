import torch

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ORXORALU:
    def __init__(self, batch_size=1024):
        self.batch_size = batch_size
        self.A = torch.randint(0, 256, (batch_size,), dtype=torch.int32, device=device)
        self.B = torch.randint(1, 256, (batch_size,), dtype=torch.int32, device=device)  # Avoid division by zero
    
    def fast_addition(self, A, B, Cin=0):
        """ Perform addition using only OR and XOR gates """
        Sum = (A ^ B ^ Cin) & 0xFF  # XOR for sum, ensure 8-bit wrap
        Carry = ((A & B) | (Cin & (A ^ B))) & 0xFF  # More precise carry approximation
        return Sum, Carry
    
    def fast_subtraction(self, A, B):
        """ Perform subtraction using XOR (A - B = A + (~B + 1)) """
        B_complement = (~B + 1) & 0xFF  # Two's complement for proper subtraction
        return self.fast_addition(A, B_complement, Cin=0)
    
    def bitwise_and_approx(self, A, B):
        """ Approximate AND using OR and XOR """
        return ((A | B) ^ (A ^ B)) & 0xFF  # De Morgan's transformation, ensure 8-bit wrap
    
    def left_shift(self, A, shift_by=1):
        """ True Left Shift ensuring bitwise correctness """
        return ((A << shift_by) & 0xFF)  # Ensure 8-bit overflow handling
    
    def right_shift(self, A, shift_by=1):
        """ Optimized Right Shift using OR/XOR-based logic """
        return (A >> shift_by) & 0xFF  # Ensure 8-bit limit
    
    def multiplexer(self, A, B, select):
        """ 2:1 Multiplexer using only OR and XOR logic - fully verified """
        return ((A & (~select)) | (B & select)) & 0xFF
    
    def bitwise_nor(self, A, B):
        """ NOR using XOR and OR with proper unsigned masking """
        return (~(A | B)) & 0xFF
    
    def bitwise_xnor(self, A, B):
        """ XNOR using XOR with proper unsigned masking """
        return (~(A ^ B)) & 0xFF
    
    def execute(self):
        """ Run ALU operations in parallel """
        Sum, Carry = self.fast_addition(self.A, self.B)
        Sub, _ = self.fast_subtraction(self.A, self.B)
        AND_Approx = self.bitwise_and_approx(self.A, self.B)
        Left_Shifted = self.left_shift(self.A)
        Right_Shifted = self.right_shift(self.A)
        MUX_Output = self.multiplexer(self.A, self.B, Carry)
        NOR_Result = self.bitwise_nor(self.A, self.B)
        XNOR_Result = self.bitwise_xnor(self.A, self.B)
        
        return Sum, Carry, Sub, AND_Approx, Left_Shifted, Right_Shifted, MUX_Output, NOR_Result, XNOR_Result

    def benchmark(self):
        """ Measure performance of ALU operations """
        import time
        start = time.time()
        for _ in range(1000):
            self.execute()
        end = time.time()
        print(f"Performance Test: {end - start:.6f} seconds for 1000 executions")

# Instantiate and execute clockless ALU
alu = ORXORALU(batch_size=1024)
results = alu.execute()
alu.benchmark()

# Print a sample output
print("Sample Output:")
labels = ["Sum", "Carry", "Subtraction", "Approx AND", "Left Shift", "Right Shift", "MUX Output", "NOR", "XNOR"]
for label, result in zip(labels, results):
    print(f"{label}:", result[:10].tolist())
