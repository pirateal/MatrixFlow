import torch
import time

class MatrixGPU:
    def __init__(self, num_cores=16, matrix_size=1024):
        self.num_cores = num_cores
        self.matrix_size = matrix_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize registers (all using int32)
        self.registers = {reg: torch.zeros(num_cores, dtype=torch.int32, device=self.device) for reg in ['EAX', 'EBX', 'ECX', 'EDX', 'ESI', 'EDI', 'EBP']}
        self.registers['ESP'] = torch.full((num_cores,), 1023, dtype=torch.int32, device=self.device)
        
        # Flags: Zero Flag (ZF), Carry Flag (CF), Overflow Flag (OF)
        self.eflags = torch.zeros(num_cores, dtype=torch.int32, device=self.device)
        
        # Memory and Stack
        self.memory = torch.zeros((num_cores, matrix_size), dtype=torch.int32, device=self.device)
        self.stack = torch.zeros((num_cores, 1024), dtype=torch.int32, device=self.device)
        
        # Execution state
        self.pc = torch.zeros(num_cores, dtype=torch.int32, device=self.device)
        self.cycle_count = torch.zeros(num_cores, dtype=torch.int32, device=self.device)
        self.halted = torch.zeros(num_cores, dtype=torch.bool, device=self.device)
        self.pc_modified = torch.zeros(num_cores, dtype=torch.bool, device=self.device)
    
    def run_program(self, program):
        start_time = time.time()
        program_length = len(program)
        
        while not torch.all(self.halted):
            active_cores = ~self.halted & (self.pc < program_length)
            if not active_cores.any():
                break
            
            current_pcs = self.pc[active_cores]
            unique_pcs = torch.unique(current_pcs)
            
            for pc_val in unique_pcs:
                cores_at_pc = active_cores & (self.pc == pc_val)
                if not cores_at_pc.any():
                    continue
                instr = program[pc_val.item()]
                self._execute_instruction_vectorized(cores_at_pc, instr)
            
            self.pc[active_cores & ~self.pc_modified] += 1
            self.pc_modified[active_cores] = False  
            self.cycle_count[active_cores] += 1
        
        print(f"Total execution time: {(time.time()-start_time)*1000:.2f}ms")
        print("Cycle counts:", self.cycle_count.cpu().numpy())
    
    def _execute_instruction_vectorized(self, core_mask, instr):
        op = instr[0]
        core_indices = core_mask.nonzero().squeeze()
        
        def get_val(operand):
            if isinstance(operand, str):
                return self.registers[operand][core_mask]
            return torch.tensor(operand, dtype=torch.int32, device=self.device).expand_as(core_indices)
        
        def set_flags(result):
            self.eflags[core_mask] = (self.eflags[core_mask] & ~(1 << 6)) | ((result == 0).int() << 6)
        
        if op == 'MOV':
            self.registers[instr[1]][core_mask] = get_val(instr[2])
        elif op == 'ADD':
            res = self.registers[instr[1]][core_mask] + get_val(instr[2])
            self.registers[instr[1]][core_mask] = res
            set_flags(res)
        elif op == 'SUB':
            res = self.registers[instr[1]][core_mask] - get_val(instr[2])
            self.registers[instr[1]][core_mask] = res
            set_flags(res)
        elif op == 'MUL':
            res = self.registers[instr[1]][core_mask] * get_val(instr[2])
            self.registers[instr[1]][core_mask] = res
            set_flags(res)
        elif op == 'DIV':
            divisor = get_val(instr[2])
            res = self.registers[instr[1]][core_mask] // divisor
            self.registers[instr[1]][core_mask] = res
            set_flags(res)
        elif op == 'AND':
            res = self.registers[instr[1]][core_mask] & get_val(instr[2])
            self.registers[instr[1]][core_mask] = res
            set_flags(res)
        elif op == 'OR':
            res = self.registers[instr[1]][core_mask] | get_val(instr[2])
            self.registers[instr[1]][core_mask] = res
            set_flags(res)
        elif op == 'XOR':
            res = self.registers[instr[1]][core_mask] ^ get_val(instr[2])
            self.registers[instr[1]][core_mask] = res
            set_flags(res)
        elif op == 'NOT':
            res = ~self.registers[instr[1]][core_mask]
            self.registers[instr[1]][core_mask] = res
            set_flags(res)
        elif op == 'SHL':
            res = self.registers[instr[1]][core_mask] << get_val(instr[2])
            self.registers[instr[1]][core_mask] = res
            set_flags(res)
        elif op == 'SHR':
            res = self.registers[instr[1]][core_mask] >> get_val(instr[2])
            self.registers[instr[1]][core_mask] = res
            set_flags(res)
        elif op == 'INC':
            res = self.registers[instr[1]][core_mask] + 1
            self.registers[instr[1]][core_mask] = res
            set_flags(res)
        elif op == 'DEC':
            res = self.registers[instr[1]][core_mask] - 1
            self.registers[instr[1]][core_mask] = res
            set_flags(res)
        elif op == 'PUSH':
            self.stack[core_mask, self.registers['ESP'][core_mask]] = get_val(instr[1])
            self.registers['ESP'][core_mask] -= 1
        elif op == 'POP':
            self.registers['ESP'][core_mask] += 1
            self.registers[instr[1]][core_mask] = self.stack[core_mask, self.registers['ESP'][core_mask]]
        elif op == 'CALL':
            self.stack[core_mask, self.registers['ESP'][core_mask]] = self.pc[core_mask]
            self.registers['ESP'][core_mask] -= 1
            self.pc[core_mask] = instr[1]
        elif op == 'RET':
            self.registers['ESP'][core_mask] += 1
            self.pc[core_mask] = self.stack[core_mask, self.registers['ESP'][core_mask]]
        elif op == 'CMP':
            set_flags(get_val(instr[1]) - get_val(instr[2]))
        elif op == 'JZ':
            self.pc[core_mask & (self.eflags[core_mask] & (1 << 6) != 0)] = instr[1]
            self.pc_modified[core_mask] = True
        elif op == 'JNZ':
            self.pc[core_mask & (self.eflags[core_mask] & (1 << 6) == 0)] = instr[1]
            self.pc_modified[core_mask] = True
        elif op == 'JMP':
            self.pc[core_mask] = instr[1]
            self.pc_modified[core_mask] = True
        elif op == 'HALT':
            self.halted[core_mask] = True

# Example program
program = [
    ('MOV', 'EAX', 10),
    ('MOV', 'EBX', 5),
    ('ADD', 'EAX', 'EBX'),  # EAX = 15
    ('SHL', 'EAX', 1),  # EAX = 30
    ('PUSH', 'EAX'),  # Push EAX onto the stack
    ('MOV', 'EAX', 50),
    ('POP', 'ECX'),  # Pop the value back into ECX
    ('HALT',)
]

gpu_cpu = MatrixGPU(num_cores=16)
gpu_cpu.run_program(program)

print("\nFinal Register Values:")
for reg, value in gpu_cpu.registers.items():
    print(f"{reg}: {value.cpu().numpy()}")
