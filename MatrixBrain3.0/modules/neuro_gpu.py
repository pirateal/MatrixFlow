import torch
import torch.nn.functional as F
import numpy as np

class NeuroGPU:
    def __init__(self, device, width=640, height=480, channels=3):
        self.device = device
        self.pixel_matrix = torch.rand((channels, height, width), device=device)

    def apply_effect(self):
        kernel = torch.tensor([[[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]]], dtype=torch.float32, device=self.device)
        input_tensor = self.pixel_matrix.unsqueeze(0)
        result = F.conv2d(input_tensor, kernel.repeat(input_tensor.shape[1], 1, 1, 1), padding=1, groups=input_tensor.shape[1])
        self.pixel_matrix = torch.clamp(result.squeeze(0), 0, 1)

    def update_frame(self):
        self.pixel_matrix = torch.rand_like(self.pixel_matrix)