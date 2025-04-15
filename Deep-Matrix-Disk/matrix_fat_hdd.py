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
            print("Loading disk from virtual.matrixdisk...")
            with open(self.disk_path, "rb") as f:
                raw = f.read()
            byte_tensor = torch.tensor(list(raw), dtype=torch.uint8, device=self.device)
            total_bytes = TOTAL_SECTORS * SECTOR_SIZE
            if len(byte_tensor) < total_bytes:
                byte_tensor = torch.cat([byte_tensor, torch.zeros(total_bytes - len(byte_tensor), dtype=torch.uint8, device=self.device)])
            self.disk = byte_tensor[:total_bytes].view(self.disk_shape)
        else:
            print("Creating new disk file...")
            self.disk = torch.zeros(self.disk_shape, dtype=torch.uint8, device=self.device)
            self.flush_to_disk()

    def flush_to_disk(self):
        with open(self.disk_path, "wb") as f:
            f.write(self.disk.cpu().numpy().tobytes())
        print("Disk flushed to SSD.")

    def write_file(self, filename, data_bytes):
        print("Test 1: Writing file...")
        num_sectors_needed = (len(data_bytes) + SECTOR_SIZE - 1) // SECTOR_SIZE
        data_tensor = torch.tensor(list(data_bytes), dtype=torch.uint8, device=self.device)
        for i in range(num_sectors_needed):
            start = i * SECTOR_SIZE
            end = min((i + 1) * SECTOR_SIZE, len(data_bytes))
            self.disk[i][:end - start] = data_tensor[start:end]
        self.flush_to_disk()

    def read_file(self, num_sectors):
        print("Test 2: Reading file...")
        return self.disk[:num_sectors].flatten().cpu().numpy().tobytes()

def run_tests():
    print("--- MatrixDrive 500MB FAT Test Init ---")
    drive = MatrixMultiDisk()
    print(f"Platters            : {PLATTERS}")
    print(f"Cylinders per Platter: {CYLINDERS}")
    print(f"Sectors per Cylinder : {SECTORS}")
    print(f"Sector Size          : {SECTOR_SIZE} bytes")
    print(f"Total Capacity       : {TOTAL_SECTORS * SECTOR_SIZE / (1024 * 1024):.2f} MB\n")

    sample_data = b"This is a test file stored in the virtual matrix hard drive using GPU logic."
    drive.write_file("testfile.txt", sample_data)
    recovered = drive.read_file(1)
    print("Recovered data:", recovered[:len(sample_data)+8])

if __name__ == "__main__":
    run_tests()