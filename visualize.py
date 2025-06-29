import matplotlib.pyplot as plt
import numpy as np
from env import Connect4Env
from stable_baselines3 import PPO

import pygame

WIDTH, HEIGHT = 700, 600
CELL_SIZE = 100

COLORS = {
    0: (200, 200, 200),
    1: (255, 0, 0),
    2: (255, 255, 0),
}

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Connect 4")
font = pygame.font.SysFont("Arial", 40)

def draw_board(board):
    for r in range(6):
        for c in range(7):
            color = COLORS[board[r, c]]
            pygame.draw.circle(screen, color, (c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2), 40)
    pygame.display.flip()

model1 = PPO.load("models/connect4_PPO")
model2 = PPO.load("models/connect4_PPO")  

score1, score2 = 0, 0

while score1 < 3 and score2 < 3:
    env = Connect4Env()
    obs = env.reset()
    done = False
    reward = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        screen.fill((0, 0, 255))
        draw_board(env.board[::])
        pygame.time.delay(500)

        if env.current_player == 1:
            action, _ = model1.predict(obs, deterministic=False)
        else:
            action, _ = model2.predict(obs, deterministic=False)

        obs, reward, done, info = env.step(action)

    if reward == 1:
        if env.current_player == 2:  # player 1 just won
            score1 += 1
        else:
            score2 += 1

    screen.fill((0, 0, 0))
    draw_board(env.board[::])
    result = f"Scores - Red: {score1} | Yellow: {score2}"
    text = font.render(result, True, (255, 255, 255))
    screen.blit(text, (50, HEIGHT // 2))
    pygame.display.flip()
    pygame.time.wait(2000)

pygame.quit()
