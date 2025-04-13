from transformers import pipeline
import torch

class LanguageProcessor:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        self.model = pipeline('text-generation', model='gpt2', device=device)

    def process_text(self, text):
        output = self.model(text, max_length=30, do_sample=True)[0]['generated_text']
        return output

    def compute_energy_boost(self):
        return 0.01 * torch.rand(1).item()