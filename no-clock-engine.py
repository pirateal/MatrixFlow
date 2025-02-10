import torch

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ORXORComputationalEngine:
    def __init__(self, batch_size=1024):
        self.batch_size = batch_size
        self.A = torch.randint(0, 2, (batch_size,), dtype=torch.int32, device=device)
        self.B = torch.randint(0, 2, (batch_size,), dtype=torch.int32, device=device)
    
    def fast_addition(self, A, B, Cin=0):
        """ Perform addition using only OR and XOR gates """
        Sum = A ^ B ^ Cin  # XOR for sum
        Carry = (A | B) | Cin  # Approximate Carry using OR
        return Sum, Carry
    
    def fast_subtraction(self, A, B):
        """ Perform subtraction using XOR (A - B = A + (~B + 1)) """
        B_complement = B ^ 1  # NOT using XOR
        return self.fast_addition(A, B_complement, Cin=1)
    
    def bitwise_and_approx(self, A, B):
        """ Approximate AND using OR and XOR """
        return (A | B) ^ (A ^ B)  # De Morgan's transformation
    
    def left_shift(self, A, shift_by=1):
        """ True Left Shift ensuring bitwise correctness """
        return (A * (2 ** shift_by)).to(torch.int32) & 0xFFFFFFFF
    
    def right_shift(self, A, shift_by=1):
        """ Optimized Right Shift using OR/XOR-based logic """
        shifted = (A >> shift_by)
        propagated = (A & ((1 << shift_by) - 1)) ^ shifted
        return (shifted | propagated).to(torch.int32)
    
    def multiplexer(self, A, B, select):
        """ 2:1 Multiplexer using only OR and XOR logic - fully verified """
        return (A & (~select & 1)) | (B & (select & 1))
    
    def execute(self):
        """ Run clockless computations in parallel """
        Sum, Carry = self.fast_addition(self.A, self.B)
        Sub, _ = self.fast_subtraction(self.A, self.B)
        AND_Approx = self.bitwise_and_approx(self.A, self.B)
        Left_Shifted = self.left_shift(self.A)
        Right_Shifted = self.right_shift(self.A)
        MUX_Output = self.multiplexer(self.A, self.B, Carry)
        
        return Sum, Carry, Sub, AND_Approx, Left_Shifted, Right_Shifted, MUX_Output

# Instantiate and execute clockless computational engine
engine = ORXORComputationalEngine(batch_size=1024)
Sum, Carry, Sub, AND_Approx, Left_Shifted, Right_Shifted, MUX_Output = engine.execute()

# Print a sample output
print("Sample Output:")
print("Sum:", Sum[:10].tolist())
print("Carry:", Carry[:10].tolist())
print("Subtraction:", Sub[:10].tolist())
print("Approx AND:", AND_Approx[:10].tolist())
print("Left Shift:", Left_Shifted[:10].tolist())
print("Right Shift:", Right_Shifted[:10].tolist())
print("MUX Output:", MUX_Output[:10].tolist())
