
import time
import numpy as np

# ---------------------------
# Configuration
# ---------------------------
BATCH_SIZE = 50_000
BITS = 5  # Used for modulo operations like (2^5 = 32)

# ---------------------------
# Backend Setup: GPU (CuPy) or CPU (NumPy)
# ---------------------------
try:
    import cupy as cp
    xp = cp
    mempool = cp.get_default_memory_pool()
    device = cp.cuda.Device(0)
    is_gpu = True
    print(f"✅ CuPy enabled - Using GPU (Compute Capability: {device.compute_capability})")
except ImportError:
    xp = np
    is_gpu = False
    print("⚠️ CuPy not found - Falling back to NumPy (CPU-only mode)")

# ---------------------------
# MatrixFlow Engine Definition
# ---------------------------
class MatrixFlowEngine:
    def __init__(self):
        self.tensors = {}

    def gear_mechanism(self):
        """Simulates a gear-rack mechanism using matrix multiplication."""
        gears = xp.random.randint(0, 2, (BATCH_SIZE, 3, 3)).astype(xp.float32)
        racks = xp.random.randint(0, 2, (BATCH_SIZE, 3, 1)).astype(xp.float32)
        self.tensors['gear'] = xp.matmul(gears, racks)
        return self.tensors['gear']

    def matrix_adder(self):
        """Simulates a simple 5-bit adder (values from 0 to 31)."""
        a = xp.random.randint(0, 32, BATCH_SIZE, dtype=xp.uint32)
        b = xp.random.randint(0, 32, BATCH_SIZE, dtype=xp.uint32)
        self.tensors['adder'] = (a + b) % 32
        return self.tensors['adder']

    def fsm_simulator(self, seq_length=1000):
        """Finite State Machine simulator using XOR-style transition logic."""
        inputs = xp.random.randint(0, 2, (BATCH_SIZE, seq_length), dtype=xp.uint8)

        if is_gpu:
            fsm_kernel = cp.ElementwiseKernel(
                'raw uint8 inputs, int32 seq_length',
                'uint8 states',
                """
                int state = 0;
                for(int i=0; i<seq_length; i++) {
                    state = (state != inputs[i]);  // XOR transition
                }
                states = state;
                """,
                'fsm_kernel'
            )
            self.tensors['fsm'] = fsm_kernel(inputs, seq_length, size=BATCH_SIZE)
        else:
            # CPU fallback logic
            def fsm_cpu(inputs):
                return np.array([np.bitwise_xor.reduce(inputs[i]) for i in range(BATCH_SIZE)], dtype=np.uint8)

            self.tensors['fsm'] = fsm_cpu(inputs)

        return self.tensors['fsm']

    def clear_memory(self):
        """Frees GPU memory (if applicable) and clears tensors."""
        if is_gpu:
            mempool.free_all_blocks()
        self.tensors.clear()

# ---------------------------
# Benchmark Function
# ---------------------------
def benchmark(engine):
    results = {}

    # Gear mechanism timing
    start = time.time()
    engine.gear_mechanism()
    results['gear_time'] = time.time() - start

    # Adder throughput
    start = time.time()
    engine.matrix_adder()
    results['adder_throughput'] = BATCH_SIZE / (time.time() - start)

    # FSM throughput
    start = time.time()
    engine.fsm_simulator(seq_length=1000)
    results['fsm_throughput'] = BATCH_SIZE / (time.time() - start)

    return results

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    print("\n🔧 Initializing MatrixFlow Engine...")
    engine = MatrixFlowEngine()
    benchmarks = benchmark(engine)

    # ---------------------------
    # Display Results
    # ---------------------------
    print("\n📊 Benchmark Results:")
    print(f"{'Gear Processing':<25} {benchmarks['gear_time']:.6f} sec/batch")
    print(f"{'Adder Throughput':<25} {benchmarks['adder_throughput']:,.0f} ops/sec")
    print(f"{'FSM Throughput':<25} {benchmarks['fsm_throughput']:,.0f} transitions/sec")

    # ---------------------------
    # Validate Output
    # ---------------------------
    adder_output = engine.tensors['adder']
    print(f"Adder Output Range: {int(xp.min(adder_output))}–{int(xp.max(adder_output))} (Expected 0–31)")

    fsm_output = engine.tensors['fsm']
    print(f"FSM Unique Final States: {xp.unique(fsm_output)}")

    # ---------------------------
    # Cleanup
    # ---------------------------
    engine.clear_memory()
    print("\n🧹 Memory cleared.\n")
