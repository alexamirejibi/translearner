import gym
import numpy as np
import sys
from shorten_trajectory import resize
from transformers import DistilBertTokenizerFast, AutoConfig, TrainingArguments
from multimodal_transformers.model import DistilBertWithTabular, TabularConfig
from scipy.special import softmax
from trainer import Trainer
import logging
# logging.disable(logging.INFO)
from utils import *


class Translearner(gym.Wrapper):
    def __init__(self, env, args,
                 trajectory=np.array([], dtype=int),
                 instructions=['go left and jump over the skull',
                               'go left and jump left over the skull to reach the ladder',
                               'goes left, jumps over the skull, goes left']):
        super().__init__(env)
        print('-=-=-=Initializing TransLEARNer=-=-=-')
        self.use_lang_rewards = True if args.lang_rewards.lower() == 'true' else False
        self.time = 0
        self.total_time = 0
        self.env = env
        self.trajectory = trajectory
        self.model = self.init_translearner()
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.instructions = self.split_instructions(args.instr) if args.instr != 'none' else instructions 
        if self.use_lang_rewards:
            print('TransLEARNer: Using language rewards')
            print('Instructions:', self.instructions)
        else:
            print('TransLEARNer: Not using language rewards')
        self.highest_lang_reward = 0
        self.lang_coefficient = args.lang_coef
        self.action_words = ACTION_WORDS


    def split_instructions(self, instructions:str):
        # split instructions (separated by [SEP]) into a list of instructions
        instruction_list = []
        for instruction in instructions.split('[SEP]'):
            instruction_list.append(instruction)
        print('given instructions: ', instruction_list)
        
        
    def init_translearner(self):
        """_summary_
        """
        config = AutoConfig.from_pretrained(
            'alexamiredjibi/Multimodal-Trajectory-Classifier')
        tabular_config = TabularConfig(num_labels=5,
                               numerical_feat_dim=18,
                               **vars(data_args))
        config.tabular_config = tabular_config
        model = DistilBertWithTabular.from_pretrained(
            'alexamiredjibi/multimodal-traj-class-no-numtransform',
            config=config)
        
        return Trainer(
            model=model,
            compute_metrics=calc_classification_metrics,
            args=TrainingArguments(output_dir='misc/translearner_logs',
                                   disable_tqdm=True)
            )
        
        
    def step(self, action):
        """_summary_

        Args:
            action (_type_): _description_

        Returns:
            _type_: _description_
        """
        next_state, reward, done, info = self.env.step(action)
        self.time += 1
        
        if done:
            self.trajectory = np.array([], dtype=int)
            self.time = 0
        if action != 0:
            self.trajectory = np.append(self.trajectory, action)
        if self.time == 34 and len(self.trajectory) > 0 and self.use_lang_rewards:
            reward = self.get_lang_reward(reward)
            self.time = 0
        if len(self.trajectory) > 70:
            self.trajectory = np.array(self.trajectory[-60:], dtype=int)
        return next_state, reward, done, info


    def reset(self):
        """_summary_

        Returns:
            _type_: _description_
        """        
        self.trajectory = np.array([], dtype=int)
        self.time = 0
        return self.env.reset()
        
    def get_lang_reward(self, reward):
        """_summary_

        Args:
            reward (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        if len(self.trajectory) == 0:
            return reward
        if len(self.trajectory) > SHORT_TRAJ_LEN + 3:
            short_traj = resize(self.trajectory.tolist())
            # print(short_traj)
        else:
            short_traj = self.trajectory.copy()
            short_traj = short_traj.astype(int)
            
        str_traj_array = traj_to_words(short_traj, simple=True)
        str_traj = ', '.join(str_traj_array)
        
        # print('trajectory: ', str_traj)
        
        d = get_torch_data(self.instructions, str_traj, self.tokenizer, self.trajectory)
        preds = self.model.predict(d)[0][0]
        soft = softmax(preds)
        # combine the softmaxed arrays into a single array by averaging
        soft_av = np.round(np.average(soft, axis=0), 4)
        # soft_av = soft[0]
        pred_label = np.argmax(soft_av)
        if pred_label != 0:
            score = soft_av[1] + soft_av[2] + soft_av[3] + soft_av[4]
            reward = reward + (score * self.lang_coefficient)
            reward = min(1, reward)
            print('pred label: ', pred_label)
            print('softmaxed average:', soft_av)
            print('reward: ', reward)

        # if reward > self.highest_lang_reward:
        #     self.highest_lang_reward = rew
            #print("highest lang reward: ", self.highest_lang_reward)
        return reward


