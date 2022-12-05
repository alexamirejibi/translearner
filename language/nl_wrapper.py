import gym
import numpy as np
import sys
from shorten_trajectory import shorten_traj_recency
from transformers import pipeline

max_traj_length = 25

# def shorten_trajectory(trajectory):
#     # remove 0's from the start of the trajectory
#     while len(trajectory) > 1 and trajectory[0] == 0:
#         trajectory.pop(0)
#     # remove 0's from the end of the trajectory
#     while len(trajectory) > 1 and trajectory[-1] == 0:
#         trajectory.pop(-1)

#     if len(trajectory) <= max_traj_length:
#         return trajectory
    
#     traj_len = len(trajectory)
#     shrink_factor = max_traj_length / traj_len

#     split_sequences = []
#     tmp = []
#     for i in range(len(trajectory)):
#         if i == 0:
#             tmp.append(trajectory[i])
#         elif trajectory[i] == trajectory[i-1]:
#             tmp.append(trajectory[i])
#         elif len(tmp) == 0:
#             tmp.append(trajectory[i])
#         else:
#             split_sequences.append(tmp)
#             tmp = [trajectory[i]]
#     split_sequences.append(tmp)
    
#     new_traj = []
#     for l in split_sequences:
#         new_count = round(len(l) * shrink_factor)
#         # if new_count == 0:
#         #     new_count = 1
#         # if new_count < 2:
#         #     new_count = 0
#         new_traj.extend(l[:new_count])

#     return new_traj

class BasicWrapper(gym.Wrapper):
    def __init__(self, env, args, trajectory=[], instruction='go left', method='reverse', lang_coefficient=0.2):
        super().__init__(env)
        print('-=-=-=Initializing language environment=-=-=-')
        self.time = 0
        self.total_time = 0
        self.env = env
        self.trajectory = trajectory
        self.model = pipeline(model="alexamiredjibi/traj-classifier-recency")
        self.instruction = args.instr if args.instr != 'none' else instruction 
        self.highest_lang_reward = 0
        self.lang_coefficient = args.lang_coef
        self.action_words = ['STAND', 'JUMP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UP-RIGHT', 'UP-LEFT',
        'DOWN-RIGHT', 'DOWN-LEFT', 'JUMP UP', 'JUMP RIGHT', 'JUMP LEFT', 'JUMP DOWN', 'JUMP UP-RIGHT',
        'JUMP UP-LEFT', 'JUMP DOWN-RIGHT', 'JUMP DOWN-LEFT']
        # self.action_words = ['STAND', 'FIRE', 'RIGHT', 'LEFT', 'FIRE RIGHT', 'FIRE LEFT']

    def get_lang_reward(self, reward):
        if len(self.trajectory) == 0:
            return reward
        str_traj_array = [self.action_words[x] for x in self.trajectory if (self.trajectory.count(x) / len(self.trajectory)) > 0.1]
        str_traj = ', '.join(str_traj_array)
        #print('trajectory is: ' + str_traj)
        clas = self.model(self.instruction + '. ' + str_traj)
        if clas[0]['label'] == 'LABEL_1':
            score = clas[0]['score']
        else:
            score = 1 - clas[0]['score']
        #print('score is: ' + str(score))
        if score < 0.1:
            score = 0
        elif score > 0.85:
            score = 1
        rew = reward + score
        rew = min(1, rew)
        if rew > reward:
            print("trajectory reward: ", rew - reward)
        if rew > self.highest_lang_reward:
            self.highest_lang_reward = rew
            print("highest lang reward: ", self.highest_lang_reward)
        return rew * self.lang_coefficient

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        # modify ...
        self.trajectory.append(action)
        self.time += 1
        if len(self.trajectory) > 25:
            self.trajectory.pop(0)
            # self.trajectory = shorten_trajectory(self.trajectory)
        if self.time == 5:
            #print('time is 5')
            reward = self.get_lang_reward(reward)
            self.time = 0
        return next_state, reward, done, info



class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        # modify obs
        return obs
    
class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        #self.trajectory = trajectory
        #self.learn_model = learn_model
    
    def reward(self, rew):
        # modify rew 
        #return rew + learn_model.get_nl_reward(self.trajectory)
        return 84

# define and initialize a class called Trajectory
# class Trajectory:
#     def __init__(self, trajectory=[]):
#         self.trajectory = trajectory

#     def add_action(self, action):
#         self.trajectory.append(action)

#     def get_trajectory(self):
#         return self.trajectory
    
# class ActionWrapper(gym.ActionWrapper):
#     def __init__(self, env, trajectory=Trajectory()):
#         super().__init__(env)
#         self.trajectory = trajectory
    
#     def action(self, act):
#         self.trajectory.add_action(act)
#         return act

class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env, trajectory=[]):
        super().__init__(env)
        self.trajectory = trajectory
    
    def action(self, act):
        return act