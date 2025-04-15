import os
import torch

SECTOR_SIZE = 4096
PLATTERS = 64
CYLINDERS = 64
SECTORS = 32
TOTAL_SECTORS = PLATTERS * CYLINDERS * SECTORS

class MatrixMultiDisk:
    def __init__(self, path="virtual.matrixdisk"):
        self.disk_path = path
        self.disk_shape = (TOTAL_SECTORS, SECTOR_SIZE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_or_initialize_disk()

    def load_or_initialize_disk(self):
        if os.path.exists(self.disk_path):
            with open(self.disk_path, "rb") as f:
                raw = f.read()
            byte_tensor = torch.tensor(list(raw), dtype=torch.uint8, device=self.device)
            total_bytes = TOTAL_SECTORS * SECTOR_SIZE
            if len(byte_tensor) < total_bytes:
                byte_tensor = torch.cat([byte_tensor, torch.zeros(total_bytes - len(byte_tensor), dtype=torch.uint8, device=self.device)])
            self.data_matrix = byte_tensor[:total_bytes].view(self.disk_shape)
        else:
            print("Disk not found. Please run matrix_fat_hdd.py first.")

def demo_matrix_disk():
    print("--- MatrixDrive 500MB Test Init ---")
    disk = MatrixMultiDisk()
    print("Disk loaded. Shape:", disk.data_matrix.shape)

if __name__ == "__main__":
    demo_matrix_disk()