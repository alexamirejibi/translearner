import pickle
# from transformers import DistilBertForSequenceClassification, DistilBertConfig, Trainer, TrainingArguments, DistilBertTokenizerFast, DataCollatorWithPadding
import random
from sklearn.model_selection import train_test_split
import sys
from dataset import AnnotatedTrajectoryDataset
from datasets import load_metric
import numpy as np
from huggingface_hub import notebook_login
from shorten_trajectory import *
from parrot import Parrot
import torch
import warnings
warnings.filterwarnings("ignore")


def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

random_state(1234)

#Init models (make sure you init ONLY once if you integrate this to your code)
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False)

# phrases = ["Can you recommed some upscale restaurants in Newyork?",
#            "What are the famous places we should not miss in Russia?"
# ]


full_sent_traj_pairs = []
short_sent_traj_pairs = []
labeled = []



with open('data/full_sent_traj_pairs.pkl', 'rb') as f:
    data = pickle.load(f)

# 5 = original
# 4 = sem similarity
# 3 = 1/3 noise + correct vec
# 2 = correct vec + half noise
# 1 = random vec from data
# 0 = fully random trajectory


k = 0
# 5 
full_data = []
for i in data:
    full = [x for x in i['trajectory'] if x != 0]
    short = shorten_trajectory(full)
    if len(short) > 0:
        id = k
        action_freqs = make_action_frequency_vector(full)
        full_data.append({'id': id, 'sentence': i['sentence'], 'full_traj': full, 'short_traj': short, 'action_freqs': action_freqs, 'label': 4}) #TODO add label
        k += 1

# # 4
sem_sim = []
for i in full_data:
    print('----------------------------------------')
    print('original sentence:', i['sentence'])
    print('--------paraphrases:')
    para_phrases = parrot.augment(input_phrase=i['sentence'])
    if para_phrases != None and len(para_phrases) > 0:
        for para_phrase in para_phrases:
            print('paraphrase: ', para_phrase)
            id = k
            action_freqs = make_action_frequency_vector(i['full_traj'])
            sem_sim.append({'id': id, 'sentence': para_phrase[0], 'full_traj': i['full_traj'], 'short_traj': i['short_traj'], 'action_freqs': action_freqs, 'label': 3})
            k += 1
        

def add_noise(trajectory, noise):
    traj = trajectory.copy()
    noise = int(len(traj) * noise)
    for i in range(noise):
        traj[random.randint(0, len(traj) - 1)] = random.randint(0, 18)
    return traj

# 4
noise_30 = []
for i in full_data:
    new_traj = add_noise(i['full_traj'], 0.3)
    short = shorten_trajectory(new_traj)
    if len(short) > 0:
        id = k
        action_freqs = make_action_frequency_vector(new_traj)
        noise_30.append({'id': id, 'sentence': i['sentence'], 'full_traj': new_traj, 'short_traj': short, 'action_freqs': action_freqs, 'label':3})
        k += 1

# 2
noise_50 = []
for i in full_data:
    new_traj = add_noise(i['full_traj'], 0.5)
    short = shorten_trajectory(new_traj)
    if len(short) > 0:
        id = k
        action_freqs = make_action_frequency_vector(new_traj)
        noise_50.append({'id': id, 'sentence': i['sentence'], 'full_traj': new_traj, 'short_traj': short, 'action_freqs': action_freqs, 'label': 2})
        k += 1


# random vecs from data / 1
random_vecs_from_data = []
for i in full_data:
    new_traj = full_data[random.randint(0, len(full_data))]['full_traj']
    short = shorten_trajectory(new_traj)
    if len(short) > 0:
        id = k
        action_freqs = make_action_frequency_vector(new_traj)
        random_vecs_from_data.append({'id': id, 'sentence': i['sentence'], 'full_traj': new_traj, 'short_traj': short, 'action_freqs': action_freqs, 'label': 1})
        k += 1

# random vecs / 0
fully_random_vecs = []
for i in full_data:
    new_traj = [random.randint(0, 18) for x in range(len(i['full_traj']))]
    short = shorten_trajectory(new_traj)
    if len(short) > 0:
        id = k
        action_freqs = make_action_frequency_vector(new_traj)
        fully_random_vecs.append({'id': id, 'sentence': i['sentence'], 'full_traj': new_traj, 'short_traj': short, 'action_freqs': action_freqs, 'label': 0})
        k += 1

# append all
new_full_data = full_data + sem_sim + noise_30 + noise_50 + random_vecs_from_data + fully_random_vecs

# print 10 examples
for i in range(10):
    #print(new_full_data[random.randint(0, len(new_full_data))])
    print(len(new_full_data))

# with open('data/short_traj_pairs_5.pkl', 'rb') as f:
#     data = pickle.load(f)
#     for i in data:
#         short_sent_traj_pairs.append({'sentence': i['sentence'],
#         'trajectory': i['trajectory']})

# for i in range(len(full_sent_traj_pairs)):
#     full_data.append({'sentence': full_sent_traj_pairs[i]['sentence'],
#     'full': full_sent_traj_pairs[i]['trajectory'], 'short': short_sent_traj_pairs[i]['trajectory']})

# print(len(full_data))
# for i in range(10):
#     print(full_data[random.randint(0, len(full_data))])