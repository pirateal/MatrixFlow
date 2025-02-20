import os
import sys
import threading
import tempfile
import shutil
import time
import errno
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from tkinterdnd2 import TkinterDnD, DND_FILES, DND_ALL
import numpy as np
import cupy as cp
from tqdm import tqdm

# -------------------------------
# Memory and File Storage Functions
# -------------------------------

# Query GPU memory using cupy and reserve 80% of total memory for this application.
free_mem, total_mem = cp.cuda.Device(0).mem_info
usable_mem = total_mem * 0.8  # Use only 80% of the total GPU memory
gpu_memory_in_gb = usable_mem / (1024 ** 3)  # For informational purposes

# Define FP4 Precision (4 bits per value)
# Our method multiplies the usable memory by 32 and then divides by 8 to simulate a 4x storage capacity.
total_capacity_fp4 = (usable_mem * 32) / 8

# Chip size adjustment (for FP4 precision)
chip_shape_fp4 = (1024, 1024, 1024)  # Dimensions for each memory "chip"
ram_chips_fp4 = {}      # Dictionary to hold allocated chips
file_registry_fp4 = {}  # Dictionary to keep track of stored files (filename -> size)

def get_used_space_fp4():
    return sum(file_registry_fp4.values())

def get_free_space_fp4():
    return total_capacity_fp4 - get_used_space_fp4()

def allocate_chip_fp4(chip_id):
    if chip_id not in ram_chips_fp4:
        ram_chips_fp4[chip_id] = cp.zeros(chip_shape_fp4, dtype=cp.uint8)

def write_block_to_chip_fp4(chip_id, start_index, data_block):
    allocate_chip_fp4(chip_id)
    chip_flat = ram_chips_fp4[chip_id].reshape(-1)
    chip_flat[start_index:start_index + data_block.size] = data_block

def read_block_from_chip_fp4(chip_id, start_index, block_size):
    if chip_id not in ram_chips_fp4:
        return cp.zeros(block_size, dtype=cp.uint8)
    chip_flat = ram_chips_fp4[chip_id].reshape(-1)
    return chip_flat[start_index:start_index + block_size]

def store_file_in_memory_fp4(file_path, chunk_bytes=1024*1024):
    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path)
    
    if file_size > get_free_space_fp4():
        messagebox.showerror("Error", "Not enough memory available!")
        return 0

    file_registry_fp4[file_name] = file_size

    with open(file_path, 'rb') as f:
        chip_id = 0
        chip_offset = 0
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Storing {file_name}") as pbar:
            while True:
                chunk = f.read(chunk_bytes)
                if not chunk:
                    break
                chunk_np = np.frombuffer(chunk, dtype=np.uint8)
                chunk_cp = cp.array(chunk_np, dtype=cp.uint8)
                bytes_remaining = chunk_cp.size
                pos = 0
                while bytes_remaining > 0:
                    available = chip_shape_fp4[0] * chip_shape_fp4[1] * chip_shape_fp4[2] - chip_offset
                    to_write = min(bytes_remaining, available)
                    block = chunk_cp[pos:pos + to_write]
                    write_block_to_chip_fp4(chip_id, chip_offset, block)
                    chip_offset += to_write
                    pos += to_write
                    bytes_remaining -= to_write
                    if chip_offset >= chip_shape_fp4[0] * chip_shape_fp4[1] * chip_shape_fp4[2]:
                        chip_id += 1
                        chip_offset = 0
                pbar.update(len(chunk))
                cp.get_default_memory_pool().free_all_blocks()
    update_free_space()
    return file_size

def store_folder_in_memory_fp4(folder_path):
    for root_dir, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root_dir, file_name)
            store_file_in_memory_fp4(file_path)
    return True

def retrieve_file_from_memory_fp4(file_name, output_path):
    """Write the file from GPU memory to a given output path."""
    if file_name not in file_registry_fp4:
        messagebox.showerror("Error", "File not found in memory!")
        return
    file_size = file_registry_fp4[file_name]
    chip_id = 0
    chip_offset = 0
    with open(output_path, 'wb') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Retrieving {file_name}") as pbar:
            while file_size > 0:
                block_size = min(file_size, chip_shape_fp4[0] * chip_shape_fp4[1] * chip_shape_fp4[2] - chip_offset)
                block = read_block_from_chip_fp4(chip_id, chip_offset, block_size)
                block_np = cp.asnumpy(block)
                f.write(block_np.tobytes())
                chip_offset += block_size
                file_size -= block_size
                pbar.update(block_size)
                if chip_offset >= chip_shape_fp4[0] * chip_shape_fp4[1] * chip_shape_fp4[2]:
                    chip_id += 1
                    chip_offset = 0

def read_file_from_memory_fp4(file_name):
    """Return the contents of a stored file as bytes (for use in the virtual file system)."""
    if file_name not in file_registry_fp4:
        raise FileNotFoundError(f"{file_name} not found in memory!")
    file_size = file_registry_fp4[file_name]
    chip_id = 0
    chip_offset = 0
    data = bytearray()
    remaining = file_size
    while remaining > 0:
        block_size = min(remaining, chip_shape_fp4[0] * chip_shape_fp4[1] * chip_shape_fp4[2] - chip_offset)
        block = read_block_from_chip_fp4(chip_id, chip_offset, block_size)
        block_np = cp.asnumpy(block)
        data.extend(block_np.tobytes())
        chip_offset += block_size
        remaining -= block_size
        if chip_offset >= chip_shape_fp4[0] * chip_shape_fp4[1] * chip_shape_fp4[2]:
            chip_id += 1
            chip_offset = 0
    return bytes(data)

def update_free_space():
    free_space = get_free_space_fp4()
    print(f"Free space updated: {free_space / (1024**2):.2f} MB")
    return free_space

# -------------------------------
# GUI Application (Matrix Memory Desktop - FP4)
# -------------------------------

class MatrixMemoryAppFP4:
    def __init__(self, root):
        self.root = root
        self.root.title("Matrix Memory Desktop - FP4")
        self.root.geometry("1000x700")
        self.dragged_item = None
        self.create_menu()
        self.create_widgets()
        self.setup_drag_support()
        self.load_files_list()
        self.refresh_ui()
        self.start_auto_update()

    def create_menu(self):
        menubar = tk.Menu(self.root)

        # File Menu: For importing files/folders only.
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Import File", command=self.import_file)
        file_menu.add_command(label="Import Folder", command=self.import_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Help Menu with Enhanced About Information
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def create_widgets(self):
        # PanedWindow divides left (file list) and right (drag/drop area)
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Left Frame: File List (files stored in memory)
        self.left_frame = ttk.Frame(self.paned, width=200, relief=tk.SUNKEN)
        self.left_frame.pack(fill=tk.Y, side=tk.LEFT)
        self.file_listbox = tk.Listbox(self.left_frame, font=("Segoe UI", 10))
        self.file_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.file_listbox.bind("<Double-Button-1>", self.on_file_double_click)
        self.file_listbox.bind("<Button-3>", self.show_context_menu)
        self.paned.add(self.left_frame)

        # Right Frame: Drag-and-Drop Area for importing files/folders
        self.right_frame = ttk.Frame(self.paned, relief=tk.SUNKEN)
        self.right_frame.pack(fill=tk.BOTH, expand=True)
        self.drop_label = ttk.Label(
            self.right_frame,
            text="Drag and Drop files/folders here to store them\n\n"
                 "To retrieve a file, drag it from the file list onto your desktop.",
            font=("Segoe UI", 14),
            anchor="center"
        )
        self.drop_label.pack(expand=True, padx=10, pady=10)
        self.right_frame.drop_target_register(DND_FILES)
        self.right_frame.dnd_bind('<<Drop>>', self.on_drop)
        self.paned.add(self.right_frame)

        # Status Bar: Displays dynamic free space updates.
        self.status_bar = ttk.Label(self.root, text="Status: ", relief=tk.SUNKEN, anchor=tk.W, font=("Segoe UI", 10))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_drag_support(self):
        # Enable dragging from the listbox (for file retrieval)
        self.file_listbox.dnd_bind('<<DragInitCmd>>', self.on_drag_start)
        self.file_listbox.dnd_bind('<<DragEndCmd>>', self.on_drag_end)
        self.file_listbox.dnd_bind('<<DragEnterCmd>>', lambda e: e.action)
        self.file_listbox.dnd_bind('<<DragLeaveCmd>>', lambda e: e.action)
        self.file_listbox.dnd_bind('<<DropCmd>>', lambda e: e.action)

    def parse_drop_paths(self, data):
        if '{' in data:
            return [p.strip('{}') for p in data.split('}{')]
        else:
            return [data]

    def on_drop(self, event):
        paths = self.parse_drop_paths(event.data)
        for path in paths:
            if os.path.exists(path):
                if os.path.isdir(path):
                    threading.Thread(target=self.store_folder_thread, args=(path,), daemon=True).start()
                elif os.path.isfile(path):
                    threading.Thread(target=self.store_file_thread, args=(path,), daemon=True).start()
        self.refresh_ui()

    def refresh_ui(self):
        self.load_files_list()
        self.update_status()

    def start_auto_update(self):
        # Update the free space status every second.
        self.update_status()
        self.root.after(1000, self.start_auto_update)

    def on_drag_start(self, event):
        selection = self.file_listbox.curselection()
        if selection:
            entry = self.file_listbox.get(selection[0])
            file_name = entry.split()[0]
            self.dragged_item = file_name
            temp_dir = os.path.join(tempfile.gettempdir(), "matrix_memory_temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, file_name)
            
            # Retrieve the file into the temporary directory.
            retrieve_event = threading.Event()
            threading.Thread(
                target=self.prepare_file_for_drag,
                args=(file_name, temp_path, retrieve_event),
                daemon=True
            ).start()
            retrieve_event.wait(5)  # Wait up to 5 seconds
            
            if os.path.exists(temp_path):
                event.data = temp_path
                event.data_type = DND_FILES
                return DND_FILES
        return None

    def prepare_file_for_drag(self, file_name, temp_path, event):
        if file_name in file_registry_fp4:
            retrieve_file_from_memory_fp4(file_name, temp_path)
        event.set()

    def on_drag_end(self, event):
        self.dragged_item = None
        # Clean up temporary files.
        temp_dir = os.path.join(tempfile.gettempdir(), "matrix_memory_temp")
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting temp file {file_path}: {e}")

    def load_files_list(self):
        self.file_listbox.delete(0, tk.END)
        for file_name, size in file_registry_fp4.items():
            self.file_listbox.insert(tk.END, f"{file_name} ({size // 1024} KB)")

    def update_status(self):
        free_space = get_free_space_fp4()
        self.status_bar.config(text=f"Free Space: {free_space / (1024**2):.2f} MB")

    def import_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            threading.Thread(target=self.store_file_thread, args=(file_path,), daemon=True).start()
            self.refresh_ui()

    def import_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            threading.Thread(target=self.store_folder_thread, args=(folder_path,), daemon=True).start()
            self.refresh_ui()

    def store_file_thread(self, file_path):
        try:
            store_file_in_memory_fp4(file_path)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        self.refresh_ui()

    def store_folder_thread(self, folder_path):
        try:
            store_folder_in_memory_fp4(folder_path)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        self.refresh_ui()

    def on_file_double_click(self, event):
        selection = self.file_listbox.curselection()
        if selection:
            entry = self.file_listbox.get(selection[0])
            file_name = entry.split()[0]
            output_path = filedialog.asksaveasfilename(initialfile=file_name)
            if output_path:
                threading.Thread(target=retrieve_file_from_memory_fp4, args=(file_name, output_path), daemon=True).start()

    def show_context_menu(self, event):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Delete", command=self.delete_selected_file)
        menu.tk_popup(event.x_root, event.y_root)

    def delete_selected_file(self):
        selection = self.file_listbox.curselection()
        if selection:
            entry = self.file_listbox.get(selection[0])
            file_name = entry.split()[0]
            if messagebox.askyesno("Confirm Delete", f"Delete {file_name}?"):
                if file_name in file_registry_fp4:
                    del file_registry_fp4[file_name]
                    self.refresh_ui()

    def show_about(self):
        about_text = (
            "Matrix Memory Desktop - FP4\n\n"
            "This program simulates an ultra-fast storage system by using your GPU's memory. It reserves 80% of the GPU memory, "
            "and through FP4 (4-bit) precision conversion, it effectively multiplies that available memory by four, "
            "creating a virtual drive with enormous capacity.\n\n"
            "In GUI mode, you can import files or folders by dragging and dropping them into the right-hand area. "
            "To retrieve a file, simply drag it from the file list onto your desktop (or any folder). "
            "Double-clicking a file in the list also allows you to save it using a dialog.\n\n"
            "In mount mode (run with the argument 'mount'), the program exposes a read-only virtual drive in Windows "
            "that appears as a physical disk. This requires WinFsp and fusepy to be installed, along with administrative rights.\n\n"
            "Enjoy the speed and capacity of GPU-based storage simulation!"
        )
        messagebox.showinfo("About", about_text)

def main_fp4():
    root = TkinterDnD.Tk()
    app = MatrixMemoryAppFP4(root)
    root.mainloop()

# -------------------------------
# Virtual File System using FUSE (Read-Only)
# -------------------------------

try:
    from fuse import FUSE, FuseOSError, Operations
except ImportError:
    # If fusepy is not available, mounting functionality will not work.
    FUSE = None

class MatrixMemoryFS(Operations):
    def __init__(self):
        now = time.time()
        # Build a dictionary of file attributes from the in-memory registry.
        self.files = {'/': {
            'st_mode': (0o40755),  # Directory
            'st_nlink': 2,
            'st_size': 0,
            'st_ctime': now,
            'st_mtime': now,
            'st_atime': now,
        }}
        for file_name, size in file_registry_fp4.items():
            self.files["/" + file_name] = {
                'st_mode': (0o100644),  # Regular file with permissions rw-r--r--
                'st_nlink': 1,
                'st_size': size,
                'st_ctime': now,
                'st_mtime': now,
                'st_atime': now,
            }

    def getattr(self, path, fh=None):
        if path in self.files:
            return self.files[path]
        raise FuseOSError(errno.ENOENT)

    def readdir(self, path, fh):
        if path == '/':
            return ['.', '..'] + [name[1:] for name in self.files if name != '/' and name.startswith('/')]
        else:
            raise FuseOSError(errno.ENOENT)

    def open(self, path, flags):
        if path not in self.files:
            raise FuseOSError(errno.ENOENT)
        return 0

    def read(self, path, size, offset, fh):
        file_name = path[1:]  # remove the leading '/'
        if file_name not in file_registry_fp4:
            raise FuseOSError(errno.ENOENT)
        data = read_file_from_memory_fp4(file_name)
        return data[offset:offset+size]

# -------------------------------
# Main: Choose GUI mode or mount mode
# -------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mount":
        if FUSE is None:
            print("FUSE (fusepy) is not available. Please install it (and WinFsp) to use mount mode.")
            sys.exit(1)
        mountpoint = sys.argv[2] if len(sys.argv) > 2 else "M:\\"
        print(f"Mounting Matrix Memory as a read-only drive at {mountpoint}")
        # Note: This mount is read-only. Dynamic updates require re-mounting.
        FUSE(MatrixMemoryFS(), mountpoint, nothreads=True, foreground=True, ro=True)
    else:
        main_fp4()
