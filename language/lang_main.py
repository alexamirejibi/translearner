import preprocessing as pre
import pickle

sentence_action_pairs = []
with open('data/train_lang_data.pkl', 'rb') as f:
    data = pickle.load(f) # data[0]['sentence'] = sentence description
    clip_to_actions = pre.load_actions('data') # clip_to_actions[clip_id]: list(actions)
    for dict in data:
        clip_id = dict['clip_id']
        sentence = dict['sentence']
        actions = clip_to_actions[clip_id]
        actions = [x for x in actions if x != 0]
        sentence_action_pairs += [(sentence, actions)]
        #print(clip_id + '\t' + sentence + '\t' + str(actions))
    print(sentence_action_pairs[1])

#actions = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']

actions = ['stand', 'fire', 'up', 'move right', 'move left', 'move down', 'move up and to the right', 'move up and to the left',
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