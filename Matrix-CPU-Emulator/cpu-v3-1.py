import torch
import time

class MatrixGPU:
    def __init__(self, num_cores=16, matrix_size=1024):
        self.num_cores = num_cores
        self.matrix_size = matrix_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize registers (all int32)
        self.registers = {reg: torch.zeros(num_cores, dtype=torch.int32, device=self.device) 
                          for reg in ['EAX', 'EBX', 'ECX', 'EDX', 'ESI', 'EDI', 'EBP']}
        self.registers['ESP'] = torch.full((num_cores,), 1023, dtype=torch.int32, device=self.device)
        
        # Flags: we’ll use bit 6 for Zero Flag (ZF)
        self.eflags = torch.zeros(num_cores, dtype=torch.int32, device=self.device)
        
        # Memory and Stack
        self.memory = torch.zeros((num_cores, matrix_size), dtype=torch.int32, device=self.device)
        self.stack = torch.zeros((num_cores, 1024), dtype=torch.int32, device=self.device)
        
        # Execution state
        self.pc = torch.zeros(num_cores, dtype=torch.int32, device=self.device)
        self.cycle_count = torch.zeros(num_cores, dtype=torch.int32, device=self.device)
        self.halted = torch.zeros(num_cores, dtype=torch.bool, device=self.device)
        self.jump_occurred = torch.zeros(num_cores, dtype=torch.bool, device=self.device)
        
        # Dispatch dictionary for instructions
        self.dispatch = {
            'MOV': self._mov,
            'ADD': self._add,
            'SUB': self._sub,
            'MUL': self._mul,
            'DIV': self._div,
            'AND': self._and,
            'OR':  self._or,
            'XOR': self._xor,
            'NOT': self._not,
            'SHL': self._shl,
            'SHR': self._shr,
            'INC': self._inc,
            'DEC': self._dec,
            'PUSH': self._push,
            'POP': self._pop,
            'CALL': self._call,
            'RET': self._ret,
            'CMP': self._cmp,
            'JZ': self._jz,
            'JNZ': self._jnz,
            'JMP': self._jmp,
            'HALT': self._halt,
        }
    
    def run_program(self, program):
        """Execute the given program."""
        start_time = time.time()
        program_length = len(program)
        
        # Ensure accurate timing on GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        while not torch.all(self.halted):
            active = ~self.halted & (self.pc < program_length)
            if not active.any():
                break
            
            # Process cores grouped by their current pc
            current_pcs = self.pc[active]
            unique_pcs = torch.unique(current_pcs)
            for pc_val in unique_pcs:
                # Get the indices of cores that are active and at this pc value
                active_indices = active.nonzero().squeeze()
                mask = torch.zeros(self.num_cores, dtype=torch.bool, device=self.device)
                # Select cores with the current pc value
                indices_at_pc = active_indices[(self.pc[active_indices] == pc_val)]
                mask[indices_at_pc] = True
                
                # Fetch and execute the instruction
                instr = program[pc_val.item()]
                op = instr[0]
                if op in self.dispatch:
                    self.dispatch[op](mask, instr)
                else:
                    raise ValueError(f"Unknown instruction {op}")
            
            # Increment pc for cores that did not jump
            increment_mask = active & (~self.jump_occurred)
            self.pc[increment_mask] += 1
            self.jump_occurred[active] = False
            self.cycle_count[active] += 1
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        print(f"Total execution time: {(time.time()-start_time)*1000:.2f}ms")
        print("Cycle counts:", self.cycle_count.cpu().numpy())
    
    def get_val(self, mask, operand):
        """
        Returns a tensor corresponding to the operand for cores specified by mask.
        If operand is a register (string), returns its slice; otherwise creates a tensor of the immediate value.
        """
        if isinstance(operand, str):
            return self.registers[operand][mask]
        count = mask.sum().item()
        return torch.full((count,), operand, dtype=torch.int32, device=self.device)
    
    def set_flags(self, mask, result):
        """Set the Zero Flag (ZF) in eflags based on the result."""
        zf = (result == 0).int()
        # Clear ZF (bit 6) and then set it based on result
        self.eflags[mask] &= ~(1 << 6)
        self.eflags[mask] |= (zf << 6)
    
    # Instruction implementations:
    
    def _mov(self, mask, instr):
        dest, src = instr[1], instr[2]
        self.registers[dest][mask] = self.get_val(mask, src)
    
    def _add(self, mask, instr):
        dest, src = instr[1], instr[2]
        val = self.get_val(mask, src)
        res = self.registers[dest][mask] + val
        self.registers[dest][mask] = res
        self.set_flags(mask, res)
    
    def _sub(self, mask, instr):
        dest, src = instr[1], instr[2]
        val = self.get_val(mask, src)
        res = self.registers[dest][mask] - val
        self.registers[dest][mask] = res
        self.set_flags(mask, res)
    
    def _mul(self, mask, instr):
        dest, src = instr[1], instr[2]
        val = self.get_val(mask, src)
        res = self.registers[dest][mask] * val
        self.registers[dest][mask] = res
        self.set_flags(mask, res)
    
    def _div(self, mask, instr):
        dest, src = instr[1], instr[2]
        divisor = self.get_val(mask, src)
        if (divisor == 0).any():
            raise ZeroDivisionError("Division by zero encountered in DIV instruction")
        res = self.registers[dest][mask] // divisor
        self.registers[dest][mask] = res
        self.set_flags(mask, res)
    
    def _and(self, mask, instr):
        dest, src = instr[1], instr[2]
        val = self.get_val(mask, src)
        res = self.registers[dest][mask] & val
        self.registers[dest][mask] = res
        self.set_flags(mask, res)
    
    def _or(self, mask, instr):
        dest, src = instr[1], instr[2]
        val = self.get_val(mask, src)
        res = self.registers[dest][mask] | val
        self.registers[dest][mask] = res
        self.set_flags(mask, res)
    
    def _xor(self, mask, instr):
        dest, src = instr[1], instr[2]
        val = self.get_val(mask, src)
        res = self.registers[dest][mask] ^ val
        self.registers[dest][mask] = res
        self.set_flags(mask, res)
    
    def _not(self, mask, instr):
        dest = instr[1]
        res = ~self.registers[dest][mask]
        self.registers[dest][mask] = res
        self.set_flags(mask, res)
    
    def _shl(self, mask, instr):
        dest, shift_val = instr[1], instr[2]
        val = self.get_val(mask, shift_val)
        res = self.registers[dest][mask] << val
        self.registers[dest][mask] = res
        self.set_flags(mask, res)
    
    def _shr(self, mask, instr):
        dest, shift_val = instr[1], instr[2]
        val = self.get_val(mask, shift_val)
        res = self.registers[dest][mask] >> val
        self.registers[dest][mask] = res
        self.set_flags(mask, res)
    
    def _inc(self, mask, instr):
        dest = instr[1]
        res = self.registers[dest][mask] + 1
        self.registers[dest][mask] = res
        self.set_flags(mask, res)
    
    def _dec(self, mask, instr):
        dest = instr[1]
        res = self.registers[dest][mask] - 1
        self.registers[dest][mask] = res
        self.set_flags(mask, res)
    
    def _push(self, mask, instr):
        val = self.get_val(mask, instr[1])
        esp = self.registers['ESP'][mask]
        self.stack[mask, esp] = val
        self.registers['ESP'][mask] = esp - 1
    
    def _pop(self, mask, instr):
        self.registers['ESP'][mask] += 1
        esp = self.registers['ESP'][mask]
        self.registers[instr[1]][mask] = self.stack[mask, esp]
    
    def _call(self, mask, instr):
        esp = self.registers['ESP'][mask]
        self.stack[mask, esp] = self.pc[mask]
        self.registers['ESP'][mask] = esp - 1
        self.pc[mask] = instr[1]
        self.jump_occurred[mask] = True
    
    def _ret(self, mask, instr):
        self.registers['ESP'][mask] += 1
        esp = self.registers['ESP'][mask]
        self.pc[mask] = self.stack[mask, esp]
        self.jump_occurred[mask] = True
    
    def _cmp(self, mask, instr):
        op1 = self.get_val(mask, instr[1])
        op2 = self.get_val(mask, instr[2])
        self.set_flags(mask, op1 - op2)
    
    def _jz(self, mask, instr):
        # Jump if Zero Flag is set
        active_indices = mask.nonzero().squeeze()
        flags = self.eflags[active_indices]
        condition = ((flags & (1 << 6)) != 0)
        if condition.any():
            selected = active_indices[condition]
            self.pc[selected] = instr[1]
            self.jump_occurred[selected] = True
    
    def _jnz(self, mask, instr):
        # Jump if Zero Flag is not set
        active_indices = mask.nonzero().squeeze()
        flags = self.eflags[active_indices]
        condition = ((flags & (1 << 6)) == 0)
        if condition.any():
            selected = active_indices[condition]
            self.pc[selected] = instr[1]
            self.jump_occurred[selected] = True
    
    def _jmp(self, mask, instr):
        self.pc[mask] = instr[1]
        self.jump_occurred[mask] = True
    
    def _halt(self, mask, instr):
        self.halted[mask] = True

# Example program:
program = [
    ('MOV', 'EAX', 10),
    ('MOV', 'EBX', 5),
    ('ADD', 'EAX', 'EBX'),  # EAX = 15
    ('SHL', 'EAX', 1),      # EAX = 30
    ('PUSH', 'EAX'),        # Push EAX onto the stack
    ('MOV', 'EAX', 50),
    ('POP', 'ECX'),         # Pop the value back into ECX
    ('HALT',)
]

gpu_cpu = MatrixGPU(num_cores=16)
gpu_cpu.run_program(program)

print("\nFinal Register Values:")
for reg, value in gpu_cpu.registers.items():
    print(f"{reg}: {value.cpu().numpy()}")
