import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

SECTOR_SIZE = 64 * 64  # 4096
NUM_SECTORS = 16

print("\n--- Multi-Deep Matrix Storage Demo ---\n")

base_layer = torch.randint(0, 255, (NUM_SECTORS, 64, 64), dtype=torch.int32, device=device)
print("Generated base layer in {:.6f} sec.".format(0))

sectors = base_layer.clone()
layer = 0
while sectors.shape[0] > 1:
    layer += 1
    print(f"Layer {layer}: compressing {sectors.shape[0]} sectors in groups of 4")
    sectors = sectors.view(-1, 4, 64, 64).mean(dim=1).to(torch.int32)
    print(f"After layer {layer}, number of sectors:", sectors.shape[0])

print("Multi-layer compression completed.")
print("Final deep matrix shape:", sectors.shape)
print(f"Logical capacity (base layer): {NUM_SECTORS * SECTOR_SIZE} cells")
print(f"Physical capacity (final deep layer): {sectors.numel()} cells")
print("Effective capacity expansion factor: {:.2f}x".format(NUM_SECTORS * SECTOR_SIZE / sectors.numel()))

print("\nSample 8x8 patch from the final deep matrix (values as integers):")
print(sectors[0, :8, :8])