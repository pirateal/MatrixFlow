# MatrixDSP

This project implements a software-defined Digital Signal Processor (DSP) using GPU-accelerated matrix logic.
# MatrixDSP

MatrixDSP is a software-defined digital signal processor (DSP) that uses GPU-accelerated matrix logic to perform high-speed signal processing tasks. By leveraging CuPy and modern GPU capabilities, MatrixDSP replaces traditional DSP hardware with flexible, software-defined modules.

## Features
- **Parallelized Multiply-Accumulate (MAC) operations** using matrix multiplications.
- **Fast Fourier Transform (FFT) Unit** for time-frequency analysis.
- **Filtering Units** for FIR and IIR filtering using matrix convolution.
- **Modulation Units** implementing AM and FM modulation.
- **Adaptive AI Integration** for dynamic tuning of filter coefficients.

## Installation
1. Ensure you have Python 3 and CuPy installed.
2. Clone this repository.
3. Install any additional dependencies (e.g., `sounddevice`, `matplotlib` for examples).

## Usage
- See the examples in the `examples/` directory to test audio filtering, FFT analysis, and modulation.
- Run tests located in `src/tests/` to verify each component.

## Future Work
- Expand DSP functionalities.
- Enhance adaptive learning for real-time tuning.
- Build a database of DSP functions for custom signal processing pipelines.
