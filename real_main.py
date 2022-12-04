
import gym 
# from gym.utils.play import play
import random
import sys
sys.path.insert(0, 'language/')
import language.nl_wrapper as nlw
# import atari wrappers
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack
from language.task_wrapper import TaskWrapper
from language.tasks import *
from language.play import play
import numpy as np
from ale_py import ALEInterface
import pickle
import time 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("ALE/MontezumaRevenge-v5")
height, width, channels = env.observation_space.shape
actions = env.action_space.n

# env = AtariWrapper(env)
#env = nlw.BasicWrapper(env)
env.reset()

env = TaskWrapper(env)
task = ClimbDownRightLadder(env)

# play(env, zoom=5)
# model = PPO("CnnPolicy", env, verbose=1, device="cpu")
# model.learn(total_timesteps=25000)
# model.save("models/PPO-model-" + 'task1')

episodes = 100
time_steps = 0
num_finished = 0
for episode in range(1, episodes+1):
    task = ClimbDownRightLadder(env)
    start_lives = 6
    env.step(0)
    done = False
    score = 0 
    dead = False

    while not (done or dead):
        if env.lives < 6:
            dead = True
        env.render()
        action = env.env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
        time_steps += 1
        time.sleep(0.01)
        num_finished = (num_finished + 1) if task.finished() else num_finished
        done = task.finished() or done
        if task.finished():
            print("finished task")
            time.sleep(2)
    print('Episode:{} Score:{}'.format(episode, score))
    # print(env.agent_pos())
    # print(env.has_key())
    # print(env.room())
    print('Finished: {}/{}'.format(num_finished, episode))
print(time_steps)
env.close()
