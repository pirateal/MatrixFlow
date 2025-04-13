import torch

class NeuroInterface:
    def __init__(self, device, grid_size=128):
        self.grid = torch.rand((grid_size, grid_size), device=device)

    def clock_cycle(self):
        self.grid = torch.clamp(self.grid * 0.99 + 0.01 * torch.rand_like(self.grid), 0, 1)