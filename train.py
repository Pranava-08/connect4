from stable_baselines3 import PPO
from env import Connect4Env,SelfPlayConnect4
import os

# print(f'iter{i}')
# Load previous model as opponent if it exists
opponent_path = "models/latest_opponent"
if os.path.exists(opponent_path + ".zip"):
    opponent_model = PPO.load(opponent_path)
else:
    # Dummy model to start with
    print('no model in given path, making dummy model')
    temp_env = Connect4Env()
    opponent_model = PPO("MlpPolicy", temp_env)

# Create self-play environment
env = SelfPlayConnect4(Connect4Env(), opponent_model)

# Wrap for vectorized training
from stable_baselines3.common.env_util import DummyVecEnv
env = DummyVecEnv([lambda: env])

# Train agent
model = PPO("MlpPolicy", env, verbose=1,)
model.learn(total_timesteps=50000)

# Save model
model.save("models/latest_opponent")
