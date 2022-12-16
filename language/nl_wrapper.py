import gym
import numpy as np
import sys
from shorten_trajectory import resize
from transformers import DistilBertTokenizerFast, AutoConfig, Trainer
from multimodal_transformers.model import DistilBertWithTabular, TabularConfig
from scipy.special import softmax

from utils import *


class Translearner(gym.Wrapper):
    def __init__(self, env, args, trajectory=np.array([], dtype=int), instruction='go left and jump'):
        super().__init__(env)
        print('-=-=-=Initializing TransLEARNer=-=-=-')
        self.time = 0
        self.total_time = 0
        self.env = env
        self.trajectory = trajectory
        self.model = self.init_translearner()
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.instruction = args.instr if args.instr != 'none' else instruction 
        self.highest_lang_reward = 0
        self.lang_coefficient = args.lang_coef
        self.action_words = ACTION_WORDS


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
            'alexamiredjibi/Multimodal-Trajectory-Classifier',
            config=config)
        
        return Trainer(
            model=model,
            compute_metrics=calc_classification_metrics
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
        # print('time: ', self.time)
        # print(len(self.trajectory))
        
        if done:
            self.trajectory = np.array([], dtype=int)
            self.time = 0
        if action != 0:
            self.trajectory = np.append(self.trajectory, action)
        if self.time == 60 and len(self.trajectory) > 0:
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
        #self.time = 0
        #print('reset trajectory')
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
            print(short_traj)
        else:
            short_traj = self.trajectory.copy()
            short_traj = short_traj.astype(int)
            
        str_traj_array = [self.action_words[int(x)] for x in short_traj]
        str_traj = ', '.join(str_traj_array)
        
        d = get_torch_data(self.instruction, str_traj, self.tokenizer, self.trajectory)
        preds = self.model.predict(d).predictions[0][0]
        pred_label = np.argmax(preds)
        soft = softmax(preds)
        soft = np.round(soft, 3)
        print(soft)
        print('pred label: ', pred_label)
        score = 0 * soft[0] + 0.25 * soft[1] + 0.5 * soft[2] + 0.75 * soft[3] + 1 * soft[4]
        # if score < 0.1:
        #     score = 0
        # elif score > 0.85:
        #     score = 1
        rew = reward + (score * self.lang_coefficient)
        rew = min(1, rew)
        if rew > reward:
            print("trajectory reward: ", rew - reward)
            # print('trajectory is: ' + str_traj)
        # else:
        #   #print('NO')
        if rew > self.highest_lang_reward:
            self.highest_lang_reward = rew
            #print("highest lang reward: ", self.highest_lang_reward)
        return rew


