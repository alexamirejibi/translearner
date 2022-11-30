import preprocessing as pre
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import random
from shorten_trajectory import shorten_trajectory
import torch
import evaluate

training_data = 'data/train_lang_data.pkl'
sentence_action_pairs = []
action_words = ['STAND', 'JUMP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UP-RIGHT', 'UP-LEFT', 'DOWN-RIGHT', 'DOWN-LEFT', 'JUMP UP', 'JUMP RIGHT', 'JUMP LEFT', 'JUMP DOWN', 'JUMP UP-RIGHT', 'JUMP UP-LEFT', 'JUMP DOWN-RIGHT', 'JUMP DOWN-LEFT']

with open(training_data, 'rb') as f:
    encoded_sentence_acton_pairs = []
    decoded_sentence_acton_pairs = []
    data = pickle.load(f) # data[0]['sentence'] = sentence description
    clip_to_actions = pre.load_actions('data') # clip_to_actions[clip_id]: list(actions)
    for dict in data:
        clip_id = dict['clip_id']
        sentence = dict['sentence']
        trajectory = clip_to_actions[clip_id]
        trajectory = shorten_trajectory(trajectory)
        trajectory = [action_words[x] for x in trajectory if (trajectory.count(x) / len(trajectory)) > 0.1]
        # combine all actions into one string
        trajectory = ', '.join(trajectory)
        if len(trajectory) > 0:
            sentence_action_pairs += [{'sentence': sentence, 'trajectory': trajectory, 'clip_id': clip_id}]

        #print(clip_id + '\t' + sentence + '\t' + str(actions))
    for i in range(10):
        print(sentence_action_pairs[random.randint(0, len(sentence_action_pairs))])
    # find the max length of trajectories in sentence_action_pairs
    # max = 0
    # min = 100
    # max_traj = ''
    # for i in sentence_action_pairs:
    #     if len(i['trajectory']) > max:
    #         max = len(i['trajectory'])
    #         max_traj = i[]
    #     min = len(i['trajectory']) if len(i['trajectory']) < min else min
    # print(str(max) + '\t' + max_traj['trajectory'] + '\t' + max_traj['sentence'])
    # print(decoded_sentence_acton_pairs[1])

# save sentence_action_pairs to file

with open('data/sentence_action_pairs.pkl', 'wb') as f:
    pickle.dump(sentence_action_pairs, f)



# TODO reduce the length of action strings? 
trajectory = ['stand', 'fire', 'up', 'move right', 'move left', 'move down', 'move up and to the right', 'move up and to the left',
'move down and to the right', 'move down and to the left', 'fire up', 'fire right', 'fire left', 'fire down',
'fire up and to the right', 'fire up and to the left', 'fire down', 'fire down and to the left']

'''
0

NOOP

1

FIRE

2

UP

3

RIGHT

4

LEFT

5

DOWN

6

UPRIGHT

7

UPLEFT

8

DOWNRIGHT

9

DOWNLEFT

10

UPFIRE

11

RIGHTFIRE

12

LEFTFIRE

13

DOWNFIRE

14

UPRIGHTFIRE

15

UPLEFTFIRE

16

DOWNRIGHTFIRE

17

DOWNLEFTFIRE
'''