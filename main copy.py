import gym
# from stable_baselines3 import A2C

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import environment.nl_wrapper as nlw

#initial environment
env_name = "MontezumaRevengeNoFrameskip-v4"
# env = gym.make(env_name)
env = make_atari_env(env_name, n_envs=4, seed=0)
print(env.action_space)
#Atari preprocessing wrapper
# env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
env = nlw.RewardWrapper(env)
max = 0
for i in range(9999):
    (tmp) = env.action_space.sample()
    max = tmp if max < tmp else max
print(max)



# Frame stacking
# env = VecFrameStack(env, n_stack=4)



# model = PPO("CnnPolicy", env, verbose=1, device="cpu")
# model.learn(total_timesteps=25000)
# model.save("models/PPO-model-" + env_name)


# # model = PPO.load("models/PPO-model-" + env_name)
# evaluate_policy(model, env, n_eval_episodes=10, render=True)

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     print(rewards)
#     env.render()