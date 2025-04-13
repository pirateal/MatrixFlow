import torch

class SystolicDSP:
    def __init__(self, device, size=16):
        self.device = device
        self.array = torch.zeros((size, size), device=device)

    def shift_and_process(self, input_val):
        self.array = torch.roll(self.array, shifts=1, dims=0)
        self.array[0] = input_val
        return self.array.sum().item()