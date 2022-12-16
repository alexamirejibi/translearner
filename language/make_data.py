import preprocessing as pre
import pickle
import random
import sys
sys.path.insert(0, 'language/')
from keep_common_actions import *


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
        trajectory = keep_common_actions(trajectory)
        # trajectory = [action_words[x] for x in trajectory if (trajectory.count(x) / len(trajectory)) > 0.1 and x != 0]
        #trajectory = [action_words[x] for x in trajectory]
        # combine all actions into one string
        #trajectory = ', '.join(trajectory)
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
    #         max_traj = i
    #     min = len(i['trajectory']) if len(i['trajectory']) < min else min
    # print(str(max), max_traj['trajectory'] + '\t' + max_traj['sentence'])
    #print(decoded_sentence_acton_pairs[1])

    # find max length of trajectory:
    # max = max(map(len, [i['trajectory'] for i in sentence_action_pairs]))
    # print('max length of trajectory: ' + str(max))

with open('data/short_traj_pairs_5.pkl', 'wb') as f:
    pickle.dump(sentence_action_pairs, f)
