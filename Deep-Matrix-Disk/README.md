# Deep Matrix Disk
A GPU-accelerated virtual hard disk using matrix-based logic and multi-layer compression to simulate massive logical storage with minimal physical space.

## Features
- Matrix-driven virtual HDD using GPU acceleration (PyTorch / CUDA)
- FAT-style sector-based read/write interface
- Multi-layer matrix compression for storage multiplication
- Emulated disk structure: 64 platters × 64 cylinders × 32 sectors
- 512MB raw matrix-backed capacity with 16x logical capacity (up to 8GB)

## Files
- `matrix_fat_hdd.py`: Main interface with FAT-like API and test cases
- `matrix_hdd_simulator.py`: Disk initialization and debugging viewer
- `multi_deep_storage_test.py`: Demonstrates multi-layer matrix compression
- `virtual.matrixdisk`: Binary storage file (auto-generated)

## Getting Started
1. Run `matrix_fat_hdd.py` to create the disk and run basic FAT tests.
2. Use `multi_deep_storage_test.py` to simulate compression.
3. Explore or mount data via `matrix_hdd_simulator.py`.

## Requirements
- Python 3.8+
- PyTorch with CUDA support (GPU required)

## Author
MatrixFlow Project – Designed for simulated matrix-based logic storage