import gym

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
import nl_wrapper as nlw
# Parallel environments
#initial environment
env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=4, seed=0)
#Atari preprocessing wrapper
# env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
env = nlw.RewardWrapper(env)

#Frame stacking
env = VecFrameStack(env, n_stack=4)

model = PPO("CnnPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

# del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards)
    env.render()