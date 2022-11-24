import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_atari_env('PongNoFrameskip', n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)

model = PPO.load("ppo_cartpole")

obs = env.reset()

trajectory = np.array([])
actions = np.array([])
traj_len = 5

while True:
    for i in range (traj_len):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        trajectory = np.append(action, trajectory)
        trajectory = trajectory[:-1]
        env.render()
        # append actions to the start of trajectory and remove the last action