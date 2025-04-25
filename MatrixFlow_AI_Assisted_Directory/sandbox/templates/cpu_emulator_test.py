# CPU Emulator Test Template
from matrixflow.emulator import CPUEmulator

def run_test():
    cpu = CPUEmulator()
    program = [0x3E, 0x05,  # MVI A,5
               0x80,        # ADD B
               0x76]        # HLT
    cpu.load_program(program)
    cpu.run()
    print("Register A:", cpu.registers['A'])

if __name__ == "__main__":
    run_test()
