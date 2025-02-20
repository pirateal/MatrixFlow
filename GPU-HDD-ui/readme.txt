# Matrix Memory Desktop - FP4

Matrix Memory Desktop - FP4 is a Python application that simulates virtual memory storage using FP4 precision on a GPU. This system allows users to store and retrieve files or folders in GPU memory, with a GUI interface for easy drag-and-drop operations.

## Key Features:
- Use of GPU memory for file storage and retrieval with FP4 precision.
- Dynamic memory management with progress bars to indicate storage and retrieval operations.
- Drag-and-drop interface to import files and folders into GPU memory.
- Real-time display of free space and stored files in the GUI.
- Multi-threading for smooth background operations.

## Requirements:
- Python 3.x
- `numpy` and `cupy` (for GPU matrix operations)
- `tkinter` and `tkinterdnd2` (for GUI and drag-and-drop functionality)
- `tqdm` (for progress bars)

## How to Use:
1. Run the Python script.
2. Drag and drop files or folders into the right pane to store them in GPU memory.
3. Double-click a file in the left pane to retrieve it to your desktop.
4. Monitor the free space and stored files in the left pane and status bar.

## Installation:
To install the required dependencies:
```bash
pip install numpy cupy tkinterdnd2 tqdm
