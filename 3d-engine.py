import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *  # Added missing GLU import
import torch
import numpy as np

# Simulation parameters
NUM_PARTICLES = 15000  # Increased particle count
GRAVITY = torch.tensor([0.0, -9.8, 0.0], device='cuda')
TIME_STEP = 0.01
BOUNCE_DAMPING = 0.8

# Initialize particles
positions = torch.rand((NUM_PARTICLES, 3), device='cuda') * 1000
velocities = torch.zeros((NUM_PARTICLES, 3), device='cuda')

# Initialize Pygame and OpenGL
def init_display():
    pygame.init()
    display = (1200, 900)  # Increased display size
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    glEnable(GL_POINT_SMOOTH)
    glPointSize(3)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display[0] / display[1]), 0.1, 5000.0)
    glTranslatef(-500, -500, -1500)

def update_simulation():
    global positions, velocities
    velocities += GRAVITY * TIME_STEP  # Apply gravity
    positions += velocities * TIME_STEP  # Update positions
    
    # Collision with ground (y=0)
    below_ground = positions[:, 1] < 0
    velocities[below_ground, 1] *= -BOUNCE_DAMPING
    positions[below_ground, 1] = 0

def draw_particles():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glBegin(GL_POINTS)
    for pos in positions.cpu().numpy():
        glVertex3fv(pos)
    glEnd()
    pygame.display.flip()

def main():
    init_display()
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
        update_simulation()
        draw_particles()
        clock.tick(60)  # Maintain FPS
    pygame.quit()

if __name__ == "__main__":
    main()
