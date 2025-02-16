import torch
from unicorn import Uc, UC_ARCH_X86, UC_MODE_32

class GPU_Emulator:
    def __init__(self):
        # Initialize Unicorn for x86 emulation
        self.uc = Uc(UC_ARCH_X86, UC_MODE_32)
        # Set up registers and memory mappings using tensors
        self.registers = {
            'EAX': torch.zeros(1, dtype=torch.int32, device="cuda"),
            'EBX': torch.zeros(1, dtype=torch.int32, device="cuda"),
            'ECX': torch.zeros(1, dtype=torch.int32, device="cuda"),
            'EDX': torch.zeros(1, dtype=torch.int32, device="cuda"),
        }
        self.memory = torch.zeros(0x10000, dtype=torch.uint8, device="cuda")
        
    def run_instruction(self, instruction):
        print(f"Executing instruction: {instruction}")
        parts = instruction.split()
        opcode = parts[0]
        
        if opcode == "MOV":
            # Remove any trailing commas from the register name
            reg = parts[1].strip(',')
            value = int(parts[2], 16)  # Convert from hex
            print(f"Moving value {value} to {reg}")
            self.registers[reg] = torch.tensor([value], dtype=torch.int32, device="cuda")
            print(f"After MOV, {reg}: {self.registers[reg].item()}")
        
        elif opcode == "ADD":
            reg1 = parts[1].strip(',')
            reg2 = parts[2].strip(',')
            print(f"Before ADD, {reg1}: {self.registers[reg1].item()} | {reg2}: {self.registers[reg2].item()}")
            self.registers[reg1] = self.registers[reg1] + self.registers[reg2]
            print(f"After ADD, {reg1}: {self.registers[reg1].item()}")
        
        elif opcode == "SUB":
            reg1 = parts[1].strip(',')
            reg2 = parts[2].strip(',')
            print(f"Before SUB, {reg1}: {self.registers[reg1].item()} | {reg2}: {self.registers[reg2].item()}")
            self.registers[reg1] = self.registers[reg1] - self.registers[reg2]
            print(f"After SUB, {reg1}: {self.registers[reg1].item()}")
        
        elif opcode == "MUL":
            reg1 = parts[1].strip(',')
            reg2 = parts[2].strip(',')
            print(f"Before MUL, {reg1}: {self.registers[reg1].item()} | {reg2}: {self.registers[reg2].item()}")
            self.registers[reg1] = self.registers[reg1] * self.registers[reg2]
            print(f"After MUL, {reg1}: {self.registers[reg1].item()}")
        
        elif opcode == "AND":
            reg1 = parts[1].strip(',')
            reg2 = parts[2].strip(',')
            print(f"Before AND, {reg1}: {self.registers[reg1].item()} | {reg2}: {self.registers[reg2].item()}")
            self.registers[reg1] = torch.bitwise_and(self.registers[reg1], self.registers[reg2])
            print(f"After AND, {reg1}: {self.registers[reg1].item()}")
        
        elif opcode == "OR":
            reg1 = parts[1].strip(',')
            reg2 = parts[2].strip(',')
            print(f"Before OR, {reg1}: {self.registers[reg1].item()} | {reg2}: {self.registers[reg2].item()}")
            self.registers[reg1] = torch.bitwise_or(self.registers[reg1], self.registers[reg2])
            print(f"After OR, {reg1}: {self.registers[reg1].item()}")
        
        elif opcode == "XOR":
            reg1 = parts[1].strip(',')
            reg2 = parts[2].strip(',')
            print(f"Before XOR, {reg1}: {self.registers[reg1].item()} | {reg2}: {self.registers[reg2].item()}")
            self.registers[reg1] = torch.bitwise_xor(self.registers[reg1], self.registers[reg2])
            print(f"After XOR, {reg1}: {self.registers[reg1].item()}")
        
        elif opcode == "NOP":
            pass

    def execute_program(self, program):
        for instruction in program:
            self.run_instruction(instruction)

# Test the updated emulator
gpu_emulator = GPU_Emulator()
program = [
    "MOV EAX, 1234",  # Sets EAX to 0x1234 = 4660
    "MOV EBX, 5678",  # Sets EBX to 0x5678 = 22136
    "ADD EAX, EBX",   # EAX = EAX + EBX
    "MUL EAX, EBX",   # EAX = EAX * EBX
    "SUB EAX, EBX",   # EAX = EAX - EBX
    "AND EAX, EBX",   # EAX = EAX & EBX
    "OR EAX, EBX",    # EAX = EAX | EBX
    "XOR EAX, EBX",   # EAX = EAX ^ EBX
    "NOP"             # No operation
]
gpu_emulator.execute_program(program)

print(f"EAX: {gpu_emulator.registers['EAX'].item()}")
print(f"EBX: {gpu_emulator.registers['EBX'].item()}")
