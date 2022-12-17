import gym
import numpy as np
from shorten_trajectory import resize
from transformers import DistilBertTokenizerFast, AutoConfig, TrainingArguments
from multimodal_transformers.model import DistilBertWithTabular, TabularConfig
from scipy.special import softmax
from trainer import Trainer
from utils import *




class Translearner(gym.Wrapper):
    """ Gym Env wrapper that contains the TransLEARNer model for language rewards.

    """
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
        # self.highest_lang_reward = 0
        self.lang_coefficient = args.lang_coef
        # self.action_words = ACTION_WORDS


    def split_instructions(self, instructions:str):
        # split instructions (separated by [SEP]) into a list of instructions
        instruction_list = []
        for instruction in instructions.split(' [SEP] '):
            instruction_list.append(instruction)
        print('given instructions: ', instruction_list)
        return instruction_list
        
        
    def init_translearner(self):
        # load the model
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
        # if you're wondering why I'm using a trainer here, and also trainer.predict() to 
        # run all my predictions - I tried very hard to get the model to predict without the trainer,
        # but I couldn't figure it out. it has something to do with the multimodal transformers
        # library. other people on github have had the same problem.
        # i think this probably somewhat degrades performance vs just predicting with the model,
        # but it hasn't made a huge difference in my experiments so far.
        
        
    def step(self, action):
        """Step the environment with the given action.

        Args:
            action (_type_): Action to be executed in the environment
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
        """ Reset the environment and returns an initial observation.
        Also resets the trajectory and time.

        Returns:
            _type_: Observation of the initial state.
        """        
        self.trajectory = np.array([], dtype=int)
        self.time = 0
        return self.env.reset()
        
    def get_lang_reward(self, reward):
        """ Get the language reward for the current trajectory.

        Args:
            reward (_type_): The reward from the environment

        Returns:
            _type_: Environment reward + language reward
        """
        
        if len(self.trajectory) == 0:
            return reward
        
        if len(self.trajectory) > SHORT_TRAJ_LEN + 3:
            # resize the trajectory to be exactly 15 actions long
            short_traj = resize(self.trajectory.tolist())
        else:
            short_traj = self.trajectory.copy()
            short_traj = short_traj.astype(int)
            
        # turn the trajectory into a string of action words
        str_traj_array = traj_to_words(short_traj, simple=True)
        str_traj = ', '.join(str_traj_array)
    
        # load the trajectory into a dataset with the instructions
        # this is the unnecessary step that wouldn't be necessary if I could just predict
        # with the model directly (instead of using the trainer)
        d = get_torch_data(self.instructions, str_traj, self.tokenizer, self.trajectory)
        preds = self.model.predict(d)[0][0]
        soft = softmax(preds)
        
        # combine the softmaxed arrays into a single array by averaging
        soft_av = np.round(np.average(soft, axis=0), 4)
        pred_label = np.argmax(soft_av)
        if pred_label != 0:
            score = soft_av[1] + soft_av[2] + soft_av[3] + soft_av[4]
            reward = reward + (score * self.lang_coefficient)
            reward = min(1, reward)
            print('pred label: ', pred_label)
            print('softmaxed average:', soft_av)
            print('reward: ', reward)
            
        return reward


