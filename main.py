
import gym 
# from gym.utils.play import play
import random
import sys
sys.path.insert(0, 'language/')
import language.nl_wrapper as nlw
# import atari wrappers
from stable_baselines3.common.atari_wrappers import AtariWrapper
from language.tasks import *
import numpy as np
from stable_baselines3 import PPO
#from stable_baselines3.common.env_util import make_atari_env
from env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy

# parse arguments
import argparse
task_dict = {0: DownLadderJumpRight, 1: ClimbDownRightLadder, 2: JumpSkullReachLadder, 
            3: JumpSkullGetKey, 4: ClimbLadderGetKey, 5: ClimbDownGoRightClimbUp, 6: JumpMiddleClimbReachLeftDoor}
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=int, default=2, help='task number 0-6'+str(task_dict))
#parser.add_argument('--lang_rewards', action=argparse.BooleanOptionalAction)
parser.add_argument('--lang_rewards', type=str, default='true', help='true = use language rewards (default) - false (or anything else = don\'t use language rewards')
parser.add_argument('--timesteps', type=int, default=500000, help='number of timesteps to play')
parser.add_argument('--lang_reward_interval', type=int, default=10, help='interval for reward calculation')
parser.add_argument('--render', type=str, default='false', help='render mode - 'true' / 'false' (defaults to 'false')')
parser.add_argument('--instr', type=str, default='none', help='provide a list of instructions separated by "[SEP]" (ex. "go left and jump [SEP] jump left [SEP] left jump")')
parser.add_argument('--lang_coef', type=float, default=0.05, help='language reward coefficient (language rewards are multiplief by the coefficient)')
# save path arg
parser.add_argument('--save_folder', type=str, default='data/train_log', help='save path')
parser.add_argument('--device', type=str, default='cuda', help='save path')
parser.add_argument('--seq_model', type=str, default='alexamiredjibi/traj-classifier-recency', help='sequence classification model')
# arg for loading model
args = parser.parse_args()

print("lang rewards: ", args.lang_rewards)
print("task: ", args.task)
print("timesteps: ", args.timesteps)
print("render: ", args.render)
print("instr: ", args.instr)
print("lang_coef: ", args.lang_coef)


#log_save_path = 'data/train_log/task-{}-lang-{}.npy'.format(args.task, args.lang_rewards)
log_save_path = '{}/task-{}-lang-{}.npy'.format(args.save_folder, args.task, args.lang_rewards)
#env = gym.make("ALE/MontezumaRevenge-v5", render_mode='human')
#height, width, channels = env.observation_space.shape
#actions = env.action_space.n

# if args.render == 'true':
#     env = gym.make("ALE/MontezumaRevenge-v5", render_mode='human')
# else:
#     env = gym.make("ALE/MontezumaRevenge-v5")
# env = gym.make("ALE/MontezumaRevenge-v5")
# env = AtariWrapper(env)

env_name = "MontezumaRevengeNoFrameskip-v4"
# env = gym.make(env_name)
env = make_atari_env(env_name, seed=0, parser_args=args, save_path=log_save_path)



# if args.lang_rewards == 'true':
#     if args.instr == 'none':
#         env = nlw.BasicWrapper(env, args=args)
#     else:
#env = nlw.BasicWrapper(env, args=args)

# env = TaskWrapper(env, save_data=True, save_path=log_save_path)
# task = task_dict[args.task](env)
# env.task = task
# _ = env.reset()
#task = ClimbLadderGetKey(env)

# play(env, zoom=5)
# 2048
#env = VecFrameStack(env, n_stack=4)
# tensorboard
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="data/tensorboard/", device=args.device)
model.learn(total_timesteps=args.timesteps)
model.save("models/PPO-task-{}-lang-{}".format(args.task, args.lang_rewards))
# evaluate_policy(model, env)

# print(env.n_steps)
# print(env.successes)
# print(env.successes_array)
# env.close()

# episodes = 100
# # time_steps = 0
# # max_time = 1000
# log_interval = 1000
# num_finished = 0
# while True:
#     task.reset()
#     start_lives = task.env.lives
#     env.step(0)
#     done = False
#     score = 0 
#     dead = False

#     while not done:
#         env.render()
#         if env.lives < start_lives:
#             done = True
#         action = env.env.action_space.sample()
#         n_state, reward, done, info = env.step(action)
#         score+=reward
#         num_finished = (num_finished + 1) if task.finished() else num_finished
#         done = env.finished() or done
#         if env.finished():
#             print("finished task")
#         #print(reward)
#     # print('Episode:{} Score:{}'.format(episode, score))
#     # print(env.agent_pos())
#     # print(env.has_key())
#     # print(env.room())
#     # print('Finished: {}/{}'.format(num_finished, episode))
# print(env.n_steps)
# print(num_finished)
# env.save_data_file()
# env.close()

