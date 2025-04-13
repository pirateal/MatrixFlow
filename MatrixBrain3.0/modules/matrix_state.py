import torch
import pygame
import numpy as np

class MatrixState:
    def __init__(self, device, h=64, w=64):
        self.device = device
        self.left = torch.rand((1, h, w), dtype=torch.float32, device=device)
        self.center = torch.rand((1, h, w), dtype=torch.float32, device=device)
        self.right = torch.rand((1, h, w), dtype=torch.float32, device=device)
        self.energy = 1.0
        self.reward = 0.0

    def update(self, reward_signal):
        adjustment = (torch.rand_like(self.center) - 0.5) * (0.01 + 0.1 * (1.0 - reward_signal))
        self.center = torch.clamp(self.center + adjustment, 0, 1)
        self.left = torch.clamp(self.left + adjustment * 0.5, 0, 1)
        self.right = torch.clamp(self.right + adjustment * 0.5, 0, 1)

    def to_surface(self, upscale=4):
        array = self.center.squeeze(0).cpu().numpy()
        array = np.clip(array, 0, 1)
        array = (array * 255).astype(np.uint8)
        surface = pygame.surfarray.make_surface(np.stack((array,)*3, axis=-1))
        size = surface.get_size()
        new_size = (size[0] * upscale, size[1] * upscale)
        return pygame.transform.scale(surface, new_size)