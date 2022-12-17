import pickle
import random
# from dataset import AnnotatedTrajectoryDataset
from shorten_trajectory import *
from utils import *


full_sent_traj_pairs = []
short_sent_traj_pairs = []
labeled = []

# set random state
random.seed(10)

"""
This file is used to create a dataset given a (description, trajectory) pairs.
This dataset can be directly used for training the transformer model.

"""

def make_datapoint(traj, dp, label, lst, transcribe_short_traj=True, include_full_traj=True, traj_for_frequency=None, traj_for_short_traj=None):
    if traj_for_short_traj is not None:
        short = resize(traj_for_short_traj)
    else:
        short = resize(traj)
        
    if traj_for_frequency is not None:
        action_freqs = make_action_frequency_vector(traj_for_frequency)
    else:    
        action_freqs = make_action_frequency_vector(traj)
    
    if transcribe_short_traj:
        short = traj_to_words(short, simple=True)
        short = ', '.join(short)
        
    if len(short) > 0:
        if include_full_traj:
            lst.append({'sentence': dp['sentence'], 'full_traj': traj, 'short_traj': short, 'action_freqs': action_freqs, 'label': label})
        else:
            lst.append({'sentence': dp['sentence'], 'short_traj': short, 'action_freqs': action_freqs, 'label': label})

def add_noise(trajectory, noise):
    traj = trajectory.copy()
    noise = int(len(traj) * noise)
    for i in range(noise):
        traj[random.randint(0, len(traj) - 1)] = random.randint(1, 17)
    return traj


with open('data/full_sent_traj_pairs.pkl', 'rb') as f:
    data = pickle.load(f)


# compile original data. this is then used to make the other data points
full_data = []
for i in data:
    full = [x for x in i['trajectory'] if x != 0]
    if len(full) == 0:
        continue
    short = resize(full)
    short = [ACTION_WORDS[x] for x in short]
    short = ', '.join(short)
    if len(short) > 0:
        action_freqs = make_action_frequency_vector(full)
        full_data.append({'sentence': i['sentence'], 'full_traj': full, 'short_traj': short, 'action_freqs': action_freqs, 'label': 4})


# labels = 3, 2, 1, 0
# -----------------------------------\\\\\\\\\\\\\
noise_30 = [] # 1/3 noise + correct sent
noise_50 = [] # 1/2 noise + correct sent
noise_75 = [] # 3/4 noise + correct sent
random_traj_from_data = [] # sent + random trajectory from data
fully_random_traj = [] # sent + fully random trajectory
random_traj_noise_30 = [] # 1/3 noise + incorrect sent
random_traj_noise_50 = [] # 1/2 noise + incorrect sent
random_traj_noise_75 = [] # 3/4 noise + incorrect sent

# these two weren't used because I forgot to append them to the final list .......
correct_freqs_incorrect_short_traj = [] # sent + random trajectory from data + random frequencies
incorrect_freqs_correct_short_traj = [] # sent + random trajectory from data + random frequencies
for i in full_data:
    make_datapoint(     
        add_noise(i['full_traj'], 0.3),
        i, 3, noise_30, transcribe_short_traj=True)
    make_datapoint(
        add_noise(i['full_traj'], 0.5),
        i, 2, noise_50, transcribe_short_traj=True)
    make_datapoint(
        add_noise(i['full_traj'], 0.75),
        i, 1, noise_75, transcribe_short_traj=True)
    make_datapoint(
        full_data[random.randint(0, len(full_data) - 1)]['full_traj'],
        i, 0, random_traj_from_data, transcribe_short_traj=True)
    make_datapoint(
        [random.randint(1, 17) for k in range(68 + random.randint(-20, 20))],
        i, 0, fully_random_traj, transcribe_short_traj=True)
    make_datapoint(
        add_noise(
            full_data[random.randint(0, len(full_data) - 1)]['full_traj'], 0.3),
        i, 0, random_traj_noise_30, transcribe_short_traj=True)
    make_datapoint(
        add_noise(
            full_data[random.randint(0, len(full_data) - 1)]['full_traj'], 0.5),
        i, 0, random_traj_noise_50, transcribe_short_traj=True)
    make_datapoint(
        add_noise(
            full_data[random.randint(0, len(full_data) - 1)]['full_traj'], 0.75),
        i, 0, random_traj_noise_75, transcribe_short_traj=True)
    make_datapoint(
        i['full_traj'], i, 0, correct_freqs_incorrect_short_traj,
        transcribe_short_traj=True, traj_for_short_traj=full_data[random.randint(0, len(full_data) - 1)]['full_traj'])
    make_datapoint(
        i['full_traj'], i, 0, incorrect_freqs_correct_short_traj,
        transcribe_short_traj=True, traj_for_frequency=full_data[random.randint(0, len(full_data) - 1)]['full_traj']
    )
    

# -----------------------------------



negative_data = random_traj_from_data + fully_random_traj + random_traj_noise_30 + random_traj_noise_50 + random_traj_noise_75
positive_data = full_data + noise_30 + noise_50 + noise_75

# randomly sample negative data to match size of positive data
negative_data = random.sample(negative_data, len(positive_data))

print(len(negative_data))
new_full_data = positive_data + negative_data
print(len(new_full_data))

# count how many have each label
for i in range(5):
    print(i, len([x for x in new_full_data if x['label'] == i]))

# count average short trajectory length for each label
for i in range(5):
    avg = 0
    for j in new_full_data:
        if j['label'] == i:
            avg += len(j['short_traj'].split(', '))
    avg = avg / len([x for x in new_full_data if x['label'] == i])
    print(i, avg)

# print 10 examples
for i in range(10):
    #print(new_full_data[random.randint(0, len(new_full_data))])
    # print(len(new_full_data))
    print(new_full_data[random.randint(0, len(new_full_data))])
    print('')


# save new_full_data as pickle
with open('data/new_augmented_data_5_labels.pkl', 'wb') as f:
    pickle.dump(new_full_data, f)
    
import csv

with open('data/new_augmented_data_5_labels.pkl', 'rb') as d:
    data = pickle.load(d)


# write to csv
header = ['description', 'label', 'short_traj'] + ACTION_WORDS
data_d = []
with open('data/new_augmented_data.csv', 'w', newline='') as f:
    # create the csv writer
    writer = csv.writer(f)
    writer.writerow(header)
    for i in data:
        tmp = [i['sentence'], i['label'], i['short_traj'], i['action_freqs'][0], i['action_freqs'][1], i['action_freqs'][2], i['action_freqs'][3], i['action_freqs'][4], i['action_freqs'][5], i['action_freqs'][6], i['action_freqs'][7], i['action_freqs'][8], i['action_freqs'][9], i['action_freqs'][10], i['action_freqs'][11], i['action_freqs'][12], i['action_freqs'][13], i['action_freqs'][14], i['action_freqs'][15], i['action_freqs'][16], i['action_freqs'][17]]
        assert len(tmp) == len(header)
        writer.writerow(tmp)
