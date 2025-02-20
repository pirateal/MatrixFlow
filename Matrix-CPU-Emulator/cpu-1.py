import cupy as cp

class MatrixCPU:
    def __init__(self):
        # GPU-based registers (each as a CuPy 1-element array for simplicity)
        self.registers = {
            'EAX': cp.array([0], dtype=cp.int32),
            'EBX': cp.array([0], dtype=cp.int32),
            'ECX': cp.array([0], dtype=cp.int32),
            'EDX': cp.array([0], dtype=cp.int32),
            'ESI': cp.array([0], dtype=cp.int32),
            'EDI': cp.array([0], dtype=cp.int32),
            'EBP': cp.array([0], dtype=cp.int32),
            'ESP': cp.array([255], dtype=cp.int32)
        }
        # GPU-based memory: simulate 1024 words
        self.memory = cp.zeros(1024, dtype=cp.int32)
        self.flags = {'ZERO': False, 'CARRY': False, 'SIGN': False, 'DIV_ZERO': False}
        self.stack = []  # We'll simulate the stack in host memory for simplicity

    def execute_program(self, program):
        print("Starting program execution...\n")
        for line in program:
            line = line.strip()
            if not line:
                continue
            print(f"Executing instruction: {line}")
            self.execute_instruction(line)
        print("\nProgram execution finished.\n")
        self.print_state()

    def execute_instruction(self, instruction):
        parts = instruction.split()
        op = parts[0]

        if op == "MOV":
            self.mov(parts)
        elif op == "ADD":
            self.add(parts)
        elif op == "SUB":
            self.sub(parts)
        elif op == "PUSH":
            self.push(parts)
        elif op == "POP":
            self.pop(parts)
        elif op == "AND":
            self.and_op(parts)
        elif op == "OR":
            self.or_op(parts)
        elif op == "CMP":
            self.cmp(parts)
        elif op == "JZ":
            self.jz()
        elif op == "JNZ":
            self.jnz()
        elif op == "JG":
            self.jg()
        elif op == "MUL":
            self.mul(parts)
        elif op == "DIV":
            self.div(parts)
        elif op == "XOR":
            self.xor(parts)
        elif op == "NOT":
            self.not_op(parts)
        elif op == "SHL":
            self.shl(parts)
        elif op == "SHR":
            self.shr(parts)
        elif op == "NEG":
            self.neg(parts)
        elif op == "MOV_MEM":
            self.mov_mem(parts)
        else:
            print(f"Unknown instruction: {instruction}")

    def mov(self, parts):
        # MOV reg value
        reg = parts[1]
        value = int(parts[2])
        self.registers[reg] = cp.array([value], dtype=cp.int32)
        print(f"Executed MOV: {reg} = {value}")

    def add(self, parts):
        reg = parts[1]
        self.registers['EAX'] += self.registers[reg]
        print(f"Executed ADD: EAX = {self.registers['EAX'].get()[0]}")

    def sub(self, parts):
        reg = parts[1]
        self.registers['EBX'] -= self.registers[reg]
        print(f"Executed SUB: EBX = {self.registers['EBX'].get()[0]}")

    def push(self, parts):
        value = int(parts[1])
        # For simplicity, simulate push by writing to host stack (not on GPU)
        self.stack.append(value)
        self.registers['ESP'] -= 1
        print(f"Executed PUSH: {value} pushed to stack. ESP = {self.registers['ESP'].get()[0]}")

    def pop(self, parts):
        if self.stack:
            value = self.stack.pop()
            self.registers['ESP'] += 1
            print(f"Executed POP: {value} popped from stack. ESP = {self.registers['ESP'].get()[0]}")
        else:
            print("Stack underflow!")

    def and_op(self, parts):
        reg = parts[1]
        self.registers['EAX'] = self.registers['EAX'] & self.registers[reg]
        print(f"Executed AND: EAX = {self.registers['EAX'].get()[0]}")

    def or_op(self, parts):
        reg = parts[1]
        self.registers['EBX'] = self.registers['EBX'] | self.registers[reg]
        print(f"Executed OR: EBX = {self.registers['EBX'].get()[0]}")

    def cmp(self, parts):
        reg = parts[1]
        result = self.registers['EAX'].get()[0] - self.registers[reg].get()[0]
        self.flags['ZERO'] = (result == 0)
        self.flags['CARRY'] = (self.registers['EAX'].get()[0] < self.registers[reg].get()[0])
        self.flags['SIGN'] = (result < 0)
        print(f"Executed CMP: {self.registers['EAX'].get()[0]} - {self.registers[reg].get()[0]} = {result} "
              f"(ZERO: {self.flags['ZERO']}, CARRY: {self.flags['CARRY']}, SIGN: {self.flags['SIGN']})")

    def jz(self):
        if self.flags['ZERO']:
            print("Executed JZ: Jump taken (ZERO flag set).")
        else:
            print("Executed JZ: No jump (ZERO flag not set).")

    def jnz(self):
        if not self.flags['ZERO']:
            print("Executed JNZ: Jump taken (ZERO flag not set).")
        else:
            print("Executed JNZ: No jump (ZERO flag set).")

    def jg(self):
        # Jump if EAX > EBX (signed comparison)
        if self.registers['EAX'].get()[0] > self.registers['EBX'].get()[0]:
            print("Executed JG: Jump taken (EAX > EBX).")
        else:
            print("Executed JG: No jump (EAX <= EBX).")

    def mul(self, parts):
        if len(parts) < 2:
            print("Error: MUL requires a register operand!")
            return
        reg = parts[1]
        self.registers['EAX'] = self.registers['EAX'] * self.registers[reg]
        print(f"Executed MUL: EAX = {self.registers['EAX'].get()[0]}")

    def div(self, parts):
        if len(parts) < 2 or self.registers[parts[1]].get()[0] == 0:
            self.flags['DIV_ZERO'] = True
            self.registers['EAX'] = cp.array([0xFFFFFFFF], dtype=cp.int32)
            print("Executed DIV: Division by zero error!")
        else:
            self.registers['EAX'] = self.registers['EAX'] // self.registers[parts[1]]
            print(f"Executed DIV: EAX = {self.registers['EAX'].get()[0]}")

    def xor(self, parts):
        reg = parts[1]
        self.registers['EAX'] = self.registers['EAX'] ^ self.registers[reg]
        print(f"Executed XOR: EAX = {self.registers['EAX'].get()[0]}")

    def not_op(self, parts):
        reg = parts[1]
        self.registers[reg] = ~self.registers[reg]
        print(f"Executed NOT: {reg} = {self.registers[reg].get()[0]}")

    def shl(self, parts):
        self.registers['EAX'] = self.registers['EAX'] << 1
        print(f"Executed SHL: EAX = {self.registers['EAX'].get()[0]}")

    def shr(self, parts):
        self.registers['EBX'] = self.registers['EBX'] >> 1
        print(f"Executed SHR: EBX = {self.registers['EBX'].get()[0]}")

    def neg(self, parts):
        self.registers['EAX'] = -self.registers['EAX']
        print(f"Executed NEG: EAX = {self.registers['EAX'].get()[0]}")

    def mov_mem(self, parts):
        # Determine if it's a memory read or write by checking for square brackets
        if parts[1].startswith('[') and parts[1].endswith(']'):
            # This is a memory read: MOV_MEM [address] reg
            address = int(parts[1][1:-1])
            reg = parts[2]
            self.registers[reg] = cp.array([int(self.memory[address].get())], dtype=cp.int32)
            print(f"Executed MOV_MEM: {reg} loaded from memory[{address}] = {self.registers[reg].get()[0]}")
        else:
            # This is a memory write: MOV_MEM address reg
            address = int(parts[1])
            reg = parts[2]
            self.memory[address] = self.registers[reg]
            print(f"Executed MOV_MEM: Memory[{address}] = {self.registers[reg].get()[0]}")

    def print_state(self):
        print("\nFinal CPU State:")
        print("Registers:", {k: int(v.get()[0]) for k, v in self.registers.items()})
        print("Flags:", self.flags)
        print("Stack:", self.stack)
        print("Memory (first 128 words):", self.memory[:128].get())

# Sample program with extended instructions
program = [
    "MOV EAX 5",           # EAX = 5
    "MOV EBX 10",          # EBX = 10
    "ADD EBX",             # EAX = EAX + EBX -> 15
    "SUB EAX",             # EBX = EBX - EAX -> 10 - 15 = -5
    "PUSH 100",            # Push 100 onto stack
    "PUSH 200",            # Push 200 onto stack
    "POP",                 # Pop (should remove 200)
    "AND EBX",             # EAX = EAX AND EBX
    "OR EAX",              # EBX = EBX OR EAX
    "CMP EBX",             # Compare EAX and EBX (CMP uses one register; here we'll compare EAX with EBX)
    "JZ",                  # Jump if ZERO flag is set
    "JNZ",                 # Jump if ZERO flag is not set
    "JG",                  # Jump if EAX > EBX
    "MUL EBX",             # EAX = EAX * EBX
    "DIV EBX",             # EAX = EAX / EBX (handle division by zero gracefully)
    "XOR EAX",             # EAX = EAX XOR EBX
    "NOT EBX",             # EBX = NOT EBX
    "SHL EAX",             # EAX = EAX << 1
    "SHR EBX",             # EBX = EBX >> 1
    "NEG EAX",             # EAX = -EAX
    "MOV_MEM 100 EAX",     # Write the value of EAX to memory at address 100
    "MOV_MEM [100] EBX"    # Read memory at address 100 into EBX
]

# Run the sample program
cpu = MatrixCPU()
cpu.execute_program(program)
cpu.print_state()
