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
# from parrot import Parrot
# import torch
# import warnings
# warnings.filterwarnings("ignore")


# def random_state(seed):
#   torch.manual_seed(seed)
#   if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

# random_state(1234)

#Init models (make sure you init ONLY once if you integrate this to your code)
# parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False)

# phrases = ["Can you recommed some upscale restaurants in Newyork?",
#            "What are the famous places we should not miss in Russia?"
# ]


full_sent_traj_pairs = []
short_sent_traj_pairs = []
labeled = []



with open('data/full_sent_traj_pairs.pkl', 'rb') as f:
    data = pickle.load(f)

# 4 = original
            # 4 = sem similarity
# 3 = 1/3 noise + correct vec
# 2 = 1/2 noise + correct vec
# 
# 1 = random vec from data
# 0 = fully random trajectory


# original 4
# 1/3 noise + correct sent 3
# 1/2 noise + correct sent 2
# 3/4 noise + correct sent 1
# random vec from data 0
# 1/3 noise + incorrect sent 0
# 1/2 noise + incorrect sent 0
# fully random trajectory 0



# 4
full_data = []
for i in data:
    full = [x for x in i['trajectory'] if x != 0]
    short = shorten_trajectory(full)
    if len(short) > 0:
        action_freqs = make_action_frequency_vector(full)
        full_data.append({'sentence': i['sentence'], 'full_traj': full, 'short_traj': short, 'action_freqs': action_freqs, 'label': 4}) #TODO add label
        

# # 2
# sem_sim = []
# for i in full_data:
#     print('----------------------------------------')
#     print('original sentence:', i['sentence'])
#     print('--------paraphrases:')
#     para_phrases = parrot.augment(input_phrase=i['sentence'])
#     if para_phrases != None and len(para_phrases) > 0:
#         for para_phrase in para_phrases:
#             print('paraphrase: ', para_phrase)
#             
#             action_freqs = make_action_frequency_vector(i['full_traj'])
#             sem_sim.append({'id': id, 'sentence': para_phrase[0], 'full_traj': i['full_traj'], 'short_traj': i['short_traj'], 'action_freqs': action_freqs, 'label': 3})
#             
        

def add_noise(trajectory, noise):
    traj = trajectory.copy()
    noise = int(len(traj) * noise)
    for i in range(noise):
        traj[random.randint(0, len(traj) - 1)] = random.randint(1, 17)
    return traj

# 3 - 1/3 noise
noise_30 = []
for i in full_data:
    new_traj = add_noise(i['full_traj'], 0.3)
    short = shorten_trajectory(new_traj)
    if len(short) > 0:
        
        action_freqs = make_action_frequency_vector(new_traj)
        noise_30.append({'sentence': i['sentence'], 'full_traj': new_traj, 'short_traj': short, 'action_freqs': action_freqs, 'label':3})
        

# 2 - 1/2 noise
noise_50 = []
for i in full_data:
    new_traj = add_noise(i['full_traj'], 0.5)
    short = shorten_trajectory(new_traj)
    if len(short) > 0:
        
        action_freqs = make_action_frequency_vector(new_traj)
        noise_50.append({'sentence': i['sentence'], 'full_traj': new_traj, 'short_traj': short, 'action_freqs': action_freqs, 'label': 2})
        

# 1 - 3/4 noise
noise_75 = []
for i in full_data:
    new_traj = add_noise(i['full_traj'], 0.75)
    short = shorten_trajectory(new_traj)
    if len(short) > 0:
        
        action_freqs = make_action_frequency_vector(new_traj)
        noise_75.append({'sentence': i['sentence'], 'full_traj': new_traj, 'short_traj': short, 'action_freqs': action_freqs, 'label': 1})
        


# random vecs from data / 0, random vecs with 1/3 noise / 0, random vecs with 1/2 noise / 0, random vecs with 3/4 noise /0.
random_vecs_from_data = []
for i in full_data:
    new_traj = full_data[random.randint(0, len(full_data) - 1)]['full_traj']
    short = shorten_trajectory(new_traj)
    if len(short) > 0:
        
        action_freqs = make_action_frequency_vector(new_traj)
        random_vecs_from_data.append({'sentence': i['sentence'], 'full_traj': new_traj, 'short_traj': short, 'action_freqs': action_freqs, 'label': 0})
        

# random vecs with 1/3 noise / 0
random_vecs_30 = []
for i in full_data:
    new_traj = add_noise(full_data[random.randint(0, len(full_data) - 1)]['full_traj'], 0.3)
    short = shorten_trajectory(new_traj)
    if len(short) > 0:
        action_freqs = make_action_frequency_vector(new_traj)
        random_vecs_30.append({'sentence': i['sentence'], 'full_traj': new_traj, 'short_traj': short, 'action_freqs': action_freqs, 'label': 0})
        
# random vecs with 1/2 noise / 0
random_vecs_50 = []
for i in full_data:
    new_traj = add_noise(full_data[random.randint(0, len(full_data) - 1)]['full_traj'], 0.5)
    short = shorten_trajectory(new_traj)
    if len(short) > 0:
        action_freqs = make_action_frequency_vector(new_traj)
        random_vecs_50.append({'sentence': i['sentence'], 'full_traj': new_traj, 'short_traj': short, 'action_freqs': action_freqs, 'label': 0})

# random vecs with 3/4 noise / 0
random_vecs_75 = []
for i in full_data:
    new_traj = add_noise(full_data[random.randint(0, len(full_data) - 1)]['full_traj'], 0.75)
    short = shorten_trajectory(new_traj)
    if len(short) > 0:
        action_freqs = make_action_frequency_vector(new_traj)
        random_vecs_75.append({'sentence': i['sentence'], 'full_traj': new_traj, 'short_traj': short, 'action_freqs': action_freqs, 'label': 0})


# fully random vecs / 0
fully_random_vecs = []
for i in full_data:
    new_traj = [random.randint(1, 17) for x in range(len(i['full_traj']))]
    short = shorten_trajectory(new_traj)
    if len(short) > 0:
        action_freqs = make_action_frequency_vector(new_traj)
        fully_random_vecs.append({'sentence': i['sentence'], 'full_traj': new_traj, 'short_traj': short, 'action_freqs': action_freqs, 'label': 0})
        

# append all
label_0 = random_vecs_from_data + random_vecs_30 + random_vecs_50 + random_vecs_75 + fully_random_vecs
label_0 = [i for i in label_0 if random.randint(1, 5) > 1]

print(len(label_0))

new_full_data = full_data + noise_30 + noise_50 + noise_75 + label_0
print(len(new_full_data))


# print 10 examples
# for i in range(10):
#     #print(new_full_data[random.randint(0, len(new_full_data))])
#     # print(len(new_full_data))
#     print(new_full_data[random.randint(0, len(new_full_data))]['label'])

# save new_full_data as pickle
with open('data/augmented_data_5_labels.pkl', 'wb') as f:
    pickle.dump(new_full_data, f)



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