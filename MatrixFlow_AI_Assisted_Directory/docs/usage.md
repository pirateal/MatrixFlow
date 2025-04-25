---
title: "Usage Guide"
tags: ["usage", "installation", "demo"]
last_updated: 2025-04-25
---

# Getting Started

## Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA Toolkit
- CuPy, NumPy, SciPy

## Installation
```bash
git clone https://github.com/pirateal/MatrixFlow.git
cd MatrixFlow
pip install -r requirements.txt
```

## Running Demos
- CPU Emulator:
  ```bash
  python scripts/gpu_matrix_ops.py --demo cpu
  ```
- Tensor Clock:
  ```bash
  jupyter notebook examples/tensor_clock_demo/demo.ipynb
  ```
