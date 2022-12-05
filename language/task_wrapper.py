import gym
import sys
import random
import numpy as np
# from scipy.misc import imresize
from utils import *
from PIL import Image
from copy import deepcopy
import tensorflow as tf
from itertools import groupby
import pdb
import pickle
import torch
#sys.path.insert(0, 'data/')
from tasks import *
from transformers import pipeline


class TaskWrapper(gym.Wrapper):
    def __init__(self, env:gym.Env, save_data=True, save_path:str=""):
        super().__init__(env)
        # task_dict = {0: DownLadderJumpRight, 1: ClimbDownRightLadder, 2: JumpSkullReachLadder, 3: JumpSkullGetKey, 4: ClimbLadderGetKey, 5: ClimbDownGoRightClimbUp, 6: JumpMiddleClimbReachLeftDoor}
        # tsk = task_dict[task_id]
        # self.task = tsk
        self.task = None
        self.env = env
        self.env.reset()
        self.n_steps = 0
        self.env = env
        self.successes = 0
        self.log_interval = 1000
        self.save_data = save_data
        self.save_path = save_path
        self.start_lives = None
        # make empty 2d numpy array
        self.successes_array = np.empty((0,2), int)
        # self.trajectory = []
        # self.model = pipeline(model="alexamiredjibi/trajectory-classifier2")
        # self.instruction = instruction
        # self.highest_lang_reward = 0
        # self.lang_coefficient = lang_coefficient
        # self.action_words = ['STAND', 'JUMP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UP-RIGHT', 'UP-LEFT',
        # 'DOWN-RIGHT', 'DOWN-LEFT', 'JUMP UP', 'JUMP RIGHT', 'JUMP LEFT', 'JUMP DOWN', 'JUMP UP-RIGHT',
        # 'JUMP UP-LEFT', 'JUMP DOWN-RIGHT', 'JUMP DOWN-LEFT']
    
    def get_task(self, task_id):
        task_dict = {0: DownLadderJumpRight, 1: ClimbDownRightLadder, 2: JumpSkullReachLadder, 3: JumpSkullGetKey, 4: ClimbLadderGetKey, 5: ClimbDownGoRightClimbUp, 6: JumpMiddleClimbReachLeftDoor}
        return task_dict[task_id](self)
    
    def assign_task(self, task):
        self.task = task
        self.task.reset()
        self.start_lives = self.env.ale.lives()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if self.start_lives != None and self.env.ale.lives() < self.start_lives:
            next_state = self.reset()
        # modify ...q
        self.n_steps += 1
        #print(self.n_steps)
        if self.task != None and self.task.finished():
            self.successes =+ 1
            reward = max(reward, 1)
            #print('task done')
        if (self.n_steps % 10000 == 0) and self.save_data:
            a = np.array([[self.n_steps, self.successes]])
            self.successes_array = np.concatenate((self.successes_array, a), axis=0)
            print('---saved data---// n_steps: {} // successes: {} // success rate: {}'.format(self.n_steps, self.successes, self.successes/self.n_steps * 100))
            self.save_data_file()
            # print('logged data')
        return next_state, reward, done, info

    def reset(self):
        self.env.reset()
        return self.task.reset()
        
    
    def finished(self):
        return self.task.finished()
    
    def add_task(self, task):
        self.task = task(self)

    def reached_pos(self, x_, y_):
        x, y = self.agent_pos()
        return (x_ - 5 <= x <= x_ + 5) and (y_ - 5 <= y <= y_ + 5)
    
    def get_data(self):
        return self.successes_array

    def save_data_file(self):
        np.save(self.save_path, self.successes_array)
        print('saved data')

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    def new_game(self, from_random_game=False):
            self._screen = self.env.reset()
            self._step(0)
            self.initial_frame = None
            return self.screen, 0, 0, self.terminal

    # def new_random_game(self):
    #     self.new_game(True)
    #     for _ in xrange(random.randint(0, RANDOM_START - 1)):
    #         self._step(0)
    #     return self.screen, 0, 0, self.terminal

    def agent_pos(self):
        x, y = self.env.ale.getRAM()[42:44]
        return int(x), int(y)

    def skull_pos(self):
        return int(self.env.ale.getRAM()[47])

    def room(self):
        return int(self.env.ale.getRAM()[3])

    def has_key(self):
        return int(self.env.ale.getRAM()[101])

    def orb_collected(self):
        return int(self.env.ale.getRAM()[49])

    def save_state(self, filename):
            state = self.env.clone_state()
            np.save(filename, state)
            print ('File written : {}'.format(filename))

    def load_state(self, filename):
            state = np.load(filename)
            self.env.restore_state(state)
            self.step(0)

    def repeat_action(self, action, n=1):
        for _ in range(n):
            self.step(action)

    def inspect(self):
        screen = self.env.ale.getScreenRGB()
        img = Image.fromarray(screen.astype('uint8'))
        img.save('trajectory/'+str(self.n_steps)+'.png')
        if self.n_steps > 100:
            input('Done')

    def new_expt(self):
        if self.args.expt_id == 1:
            self.task = ClimbDownRightLadder(self)
        elif self.args.expt_id == 2:
            self.task = JumpSkullReachLadder(self)
        elif self.args.expt_id == 3:
            self.task = ClimbLadderGetKey(self)
        elif self.args.expt_id == 4:
            self.task = ClimbDownGoRightClimbUp(self)
        elif self.args.expt_id == 5:
            self.task = JumpMiddleClimbReachLeftDoor(self)
            
        self._step(0)
        self._step(0)
        self._step(0)
        self._step(0)
        for _ in range(random.randint(0, RANDOM_START - 1)):
            self._step(0)

        return self.screen, 0, 0, self.terminal
        

    # def _step(self, action):
    #     self._screen, self.reward, self.terminal, _ = self.env.step(action)
    #     self.n_steps += 1

    # def _random_step(self):
    #     action = self.env.action_space.sample()
    #     self._step(action)

    # @ property
    # def screen(self):
    #     return imresize(rgb2gray(self._screen)/255., self.dims)

    @property
    def action_size(self):
        return self.env.action_space.n

    @property
    def lives(self):
        return self.env.ale.lives()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def state(self):
        return self.screen, self.reward, self.terminal

    def act(self, action):
        start_lives = self.lives
        self.terminal = False
        self.action_vector[action] += 1.

        self._step(action)

        if start_lives > self.lives:
            self.terminal = True
        
        if not self.terminal:
            goal_reached = self.task.finished()
        else:
            goal_reached = False

        if goal_reached:
            self.reward = 1.0
            self.terminal = True
        else:
            self.reward = 0.0

        if self.args.lang_coeff > 0.0:
            lang_reward = self.args.lang_coeff * self.compute_language_reward()
            self.reward += lang_reward
        if self.n_steps > MAX_STEPS:
            self.terminal = True
        
        if self.terminal:
            self.reset()

        return self.state, goal_reached




# class ObservationWrapper(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
    
#     def observation(self, obs):
#         # modify obs
#         return obs
    
# class RewardWrapper(gym.RewardWrapper):
#     def __init__(self, env):
#         super().__init__(env)
    
#     def reward(self, rew):
#         # modify rew
#         return rew
    
# class ActionWrapper(gym.ActionWrapper):
#     def __init__(self, env):
#         super().__init__(env)
    
#     def action(self, act):
#         # modify act
#         return act

# env = BasicWrapper(gym.make("CartPole-v0"))
