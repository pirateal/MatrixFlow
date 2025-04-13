import pygame
import torch
import random
import numpy as np
from modules.matrix_state import MatrixState
from modules.system_state import SystemState
from modules.matrix_logic import MatrixLogic
from modules.neuro_gpu import NeuroGPU
from modules.systolic_dsp import SystolicDSP
from modules.neuro_interface import NeuroInterface
from modules.language_processor import LanguageProcessor
from modules.dream_processor import DreamProcessor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MatrixBrain:
    def __init__(self):
        self.matrix_state = MatrixState(device)
        self.system_state = SystemState()
        self.matrix_logic = MatrixLogic(device)
        self.neuro_gpu = NeuroGPU(device)
        self.systolic_dsp = SystolicDSP(device)
        self.neuro_interface = NeuroInterface(device)
        self.language_processor = LanguageProcessor()
        self.dream_processor = DreamProcessor()
        self.frame_counter = 0
        self.running = True

        pygame.init()
        self.screen = pygame.display.set_mode((1000, 800))
        pygame.display.set_caption('MatrixBrain 3.0 Turbo')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def run_frame(self):
        self.frame_counter += 1
        reward = self.matrix_logic.compute_reward_signal(self.matrix_state.center)
        self.matrix_state.update(reward)

        if self.frame_counter % 30 == 0:
            self.neuro_gpu.update_frame()
        self.neuro_gpu.apply_effect()
        self.neuro_interface.clock_cycle()

        if self.frame_counter % 300 == 0:
            txt = self.language_processor.process_text('Hello evolving matrix brain.')
            boost = self.language_processor.compute_energy_boost()
            self.matrix_state.energy = min(1.0, self.matrix_state.energy + boost)

        if self.frame_counter % 1000 == 0:
            self.dream_processor.dream(self.matrix_state, self.system_state)

        self.system_state.update_emotion(self.matrix_state.energy)
        if reward > 0.5:
            self.system_state.save_memory(self.matrix_state.center)

        self.render()

    def render(self):
        self.screen.fill((10, 10, 10))

        # Render NeuroGPU vision cortex
        video = self.neuro_gpu.pixel_matrix.cpu().permute(1, 2, 0).numpy()
        video = np.clip(video, 0, 1)
        video = (video * 255).astype(np.uint8)
        video_surface = pygame.surfarray.make_surface(np.transpose(video, (1, 0, 2)))
        video_surface = pygame.transform.scale(video_surface, (980, 400))
        self.screen.blit(video_surface, (10, 10))

        # Render Left, Center, Right matrices
        left_surface = self.matrix_state.to_surface(upscale=2)
        center_surface = self.matrix_state.to_surface(upscale=2)
        right_surface = self.matrix_state.to_surface(upscale=2)
        self.screen.blit(left_surface, (50, 450))
        self.screen.blit(center_surface, (370, 450))
        self.screen.blit(right_surface, (690, 450))

        # HUD Overlay
        text = f'Frame: {self.frame_counter} | Energy: {self.matrix_state.energy:.2f} | Reward: {self.matrix_state.reward:.2f} | Emotion: {self.system_state.emotion}'
        text_surface = self.font.render(text, True, (255, 255, 255))
        self.screen.blit(text_surface, (20, 750))

        pygame.display.flip()

    def main_loop(self):
        while self.running:
            self.handle_events()
            self.run_frame()
            self.clock.tick(30)
        pygame.quit()

if __name__ == '__main__':
    brain = MatrixBrain()
    brain.main_loop()