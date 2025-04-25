# VPU Pipeline Example
import numpy as np
from matrixflow.vpu import VectorProcessingUnit

def demo_pipeline():
    vpu = VectorProcessingUnit()
    data = np.random.rand(8)
    result = vpu.process(data)
    print("Input:", data)
    print("Output:", result)

if __name__ == "__main__":
    demo_pipeline()
