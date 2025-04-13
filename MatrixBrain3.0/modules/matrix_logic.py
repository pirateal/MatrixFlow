import torch

class MatrixLogic:
    def __init__(self, device):
        self.device = device

    def compute_reward_signal(self, matrix):
        return 1.0 - torch.abs(matrix - 0.5).mean().item()