import torch

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ORXORComputationalEngine:
    def __init__(self, batch_size=1024):
        self.batch_size = batch_size
        self.A = torch.randint(0, 2, (batch_size,), dtype=torch.int8, device=device)
        self.B = torch.randint(0, 2, (batch_size,), dtype=torch.int8, device=device)
    
    def fast_addition(self, A, B, Cin=0):
        """ Perform addition using only OR and XOR gates """
        Sum = A ^ B ^ Cin  # XOR for sum
        Carry = (A | B) | Cin  # Approximate Carry using OR
        return Sum, Carry
    
    def multiplexer(self, A, B, select):
        """ 2:1 Multiplexer using OR and XOR """
        return (A ^ B) ^ (select * (A | B))
    
    def execute(self):
        """ Run clockless computations in parallel """
        Sum, Carry = self.fast_addition(self.A, self.B)
        MUX_Output = self.multiplexer(self.A, self.B, Carry)
        return Sum, Carry, MUX_Output

# Instantiate and execute clockless computational engine
engine = ORXORComputationalEngine(batch_size=1024)
Sum, Carry, MUX_Output = engine.execute()

# Print a sample output
print("Sample Output:")
print("Sum:", Sum[:10].tolist())
print("Carry:", Carry[:10].tolist())
print("MUX Output:", MUX_Output[:10].tolist())
