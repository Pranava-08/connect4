### env/connect4_env.py
import gym
import numpy as np
from gym import spaces

class Connect4Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.seed()
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.action_space = spaces.Discrete(self.cols)
        self.observation_space = spaces.Box(0, 2, shape=(self.rows, self.cols), dtype=np.int8)
        self.current_player = 1

    # Add inside your Connect4Env class
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.board[:] = 0
        self.current_player = 1
        return self.board.copy()

    def step(self, action):
        if not 0 <= action < self.cols or self.board[0][action] != 0:
            return self.board.copy(), -10, True, {"invalid": True}

        for row in reversed(range(self.rows)):
            if self.board[row][action] == 0:
                self.board[row][action] = self.current_player
                break

        done, reward = self.check_done()
        obs = self.board.copy()
        self.current_player = 3 - self.current_player
        return obs, reward, done, {}

    def check_done(self):
        board = self.board
        for c in range(self.cols - 3):
            for r in range(self.rows):
                if board[r, c] == self.current_player and all(board[r, c+i] == self.current_player for i in range(4)):
                    return True, 1

        for c in range(self.cols):
            for r in range(self.rows - 3):
                if board[r, c] == self.current_player and all(board[r+i, c] == self.current_player for i in range(4)):
                    return True, 1

        for c in range(self.cols - 3):
            for r in range(self.rows - 3):
                if board[r, c] == self.current_player and all(board[r+i, c+i] == self.current_player for i in range(4)):
                    return True, 1

        for c in range(self.cols - 3):
            for r in range(3, self.rows):
                if board[r, c] == self.current_player and all(board[r-i, c+i] == self.current_player for i in range(4)):
                    return True, 1

        if np.all(board != 0):
            return True, 0.5  # Draw

        return False, 0

    def render(self, mode='human'):
        print(self.board[::-1])

class SelfPlayConnect4(gym.Wrapper):
    def __init__(self, env, opponent_model):
        super().__init__(env)
        self.opponent_model = opponent_model

    def step(self, action):
        # Agent (player 1) move
        obs, reward, done, info = self.env.step(action)

        if done:
            return obs, reward, done, info

        # Opponent (player 2) move
        opponent_action, _ = self.opponent_model.predict(obs, deterministic=False)
        obs, reward, done, info = self.env.step(opponent_action)

        # Reverse reward since opponent played
        return obs, -reward, done, info
