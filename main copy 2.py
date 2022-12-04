import gym
# from stable_baselines3 import A2C

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import language.nl_wrapper as nlw
import time
import numpy as np
from stable_baselines3.common.env_util import unwrap_wrapper
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.atari_wrappers import AtariWrapper
import gym


#initial environment
#env_name = "BreakoutNoFrameskip-v4"
# env_name = "PongNoFrameskip-v4"
env_name = "MontezumaRevengeNoFrameskip-v4"
# env = gym.make(env_name)
env = make_atari_env(env_name, seed=0)
#env = VecFrameStack(env, n_stack=4)
#Atari preprocessing wrapper
# env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
# env = nlw.RewardWrapper(env)
# env = nlw.ActionWrapper(env, traj)


env = nlw.BasicWrapper(env, instruction='Climb down the ladder, go left and jump to the left platform')




# # Frame stacking
# #env = VecFrameStack(env, n_stack=4)
# model = PPO("CnnPolicy", env, verbose=1, device="cpu")
# # model.learn(total_timesteps=25000)
# # model.save("models/PPO-model-" + env_name)
# action_words = ['STAND', 'JUMP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UP-RIGHT', 'UP-LEFT', 'DOWN-RIGHT', 'DOWN-LEFT', 'JUMP UP', 'JUMP RIGHT', 'JUMP LEFT', 'JUMP DOWN', 'JUMP UP-RIGHT', 'JUMP UP-LEFT', 'JUMP DOWN-RIGHT', 'JUMP DOWN-LEFT']

# # model = PPO.load("models/PPO-model-" + env_name)
# # evaluate_policy(model, env, n_eval_episodes=10, render=True)

# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     #action = env.action_space.sample()
#     #print('1 ', type(action1))
#     obs, rewards, dones, info = env.step(action)
#     # print(rewards)
#     env.render()
#     print(rewards)
#     #time.sleep(0.1)
#     #print(action_words[action[0]], action)
# # print(trajectory.get_trajectory())
#     # print(action)