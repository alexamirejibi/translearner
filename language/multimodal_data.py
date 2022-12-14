import pickle
import random
# from dataset import AnnotatedTrajectoryDataset
from shorten_trajectory import *
from utils import action_words


full_sent_traj_pairs = []
short_sent_traj_pairs = []
labeled = []

# set random state
random.seed(10)

def make_datapoint(traj, dp, label, lst, transcribe_acts=False, include_full_traj=True):
    short = shorten_trajectory(traj)
    action_freqs = make_action_frequency_vector(traj)
    if transcribe_acts:
        short = [action_words[x] for x in short if x != 0]
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



# label = 4 (original data)
full_data = []
for i in data:
    full = [x for x in i['trajectory'] if x != 0]
    short = shorten_trajectory(full)
    short = [action_words[x] for x in short if x != 0]
    short = ', '.join(short)
    if len(short) > 0:
        action_freqs = make_action_frequency_vector(full)
        full_data.append({'sentence': i['sentence'], 'full_traj': full, 'short_traj': short, 'action_freqs': action_freqs, 'label': 4})
    
# # find average length of trajectory
# avg = 0
# for i in full_data:
#     avg += len(i['full_traj'])
# avg = avg / len(full_data)
# print('average length', avg)



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
for i in full_data:
    make_datapoint(     
            add_noise(i['full_traj'], 0.3),
        i, 3, noise_30, transcribe_acts=True)
    make_datapoint(
        add_noise(i['full_traj'], 0.5),
        i, 2, noise_50, transcribe_acts=True)
    make_datapoint(
        add_noise(i['full_traj'], 0.75),
        i, 1, noise_75, transcribe_acts=True)
    make_datapoint(
        full_data[random.randint(0, len(full_data) - 1)]['full_traj'],
        i, 0, random_traj_from_data, transcribe_acts=True)
    make_datapoint(
        [random.randint(1, 17) for k in range(68 + random.randint(-20, 20))],
        i, 0, fully_random_traj, transcribe_acts=True)
    make_datapoint(
        add_noise(
            full_data[random.randint(0, len(full_data) - 1)]['full_traj'], 0.3),
        i, 0, random_traj_noise_30, transcribe_acts=True)
    make_datapoint(
        add_noise(
            full_data[random.randint(0, len(full_data) - 1)]['full_traj'], 0.5),
        i, 0, random_traj_noise_50, transcribe_acts=True)
    make_datapoint(
        add_noise(
            full_data[random.randint(0, len(full_data) - 1)]['full_traj'], 0.75),
        i, 0, random_traj_noise_75, transcribe_acts=True)

# -----------------------------------



negative_data = random_traj_from_data + fully_random_traj + random_traj_noise_30 + random_traj_noise_50 + random_traj_noise_75
negative_data = [i for i in negative_data if random.randint(1, 5) > 1] # to make it 50/50 pos/neg

print(len(negative_data))
new_full_data = full_data + noise_30 + noise_50 + noise_75 + negative_data
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

# 0 24177
# 1 6057
# 2 6057
# 3 6057
# 4 6057

# average short traj length for each label
# 0 10.90817719320015
# 1 12.80155192339442
# 2 13.272081888723791
# 3 12.59402344394915
# 4 10.010070992240383


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
# # 3 - 1/3 noise
# noise_30 = []
# for i in full_data:
#     new_traj = add_noise(i['full_traj'], 0.3)
#     short = shorten_trajectory(new_traj)
#     if len(short) > 0:
        
#         action_freqs = make_action_frequency_vector(new_traj)
#         noise_30.append({'sentence': i['sentence'], 'full_traj': new_traj, 'short_traj': short, 'action_freqs': action_freqs, 'label':3})
        

# # 2 - 1/2 noise
# noise_50 = []
# for i in full_data:
#     new_traj = add_noise(i['full_traj'], 0.5)
#     short = shorten_trajectory(new_traj)
#     if len(short) > 0:
#         action_freqs = make_action_frequency_vector(new_traj)
#         noise_50.append({'sentence': i['sentence'], 'full_traj': new_traj, 'short_traj': short, 'action_freqs': action_freqs, 'label': 2})
        

# # 1 - 3/4 noise
# noise_75 = []
# for i in full_data:
#     new_traj = add_noise(i['full_traj'], 0.75)
#     short = shorten_trajectory(new_traj)
#     if len(short) > 0:
        
#         action_freqs = make_action_frequency_vector(new_traj)
#         noise_75.append({'sentence': i['sentence'], 'full_traj': new_traj, 'short_traj': short, 'action_freqs': action_freqs, 'label': 1})
        


# # random vecs from data / 0, random vecs with 1/3 noise / 0, random vecs with 1/2 noise / 0, random vecs with 3/4 noise /0.
# random_vecs_from_data = []
# for i in full_data:
#     new_traj = full_data[random.randint(0, len(full_data) - 1)]['full_traj']
#     short = shorten_trajectory(new_traj)
#     if len(short) > 0:
        
#         action_freqs = make_action_frequency_vector(new_traj)
#         random_vecs_from_data.append({'sentence': i['sentence'], 'full_traj': new_traj, 'short_traj': short, 'action_freqs': action_freqs, 'label': 0})
        


# # random vecs with 1/3 noise / 0


# random_vecs_30 = []
# for i in full_data:
#     new_traj = add_noise(full_data[random.randint(0, len(full_data) - 1)]['full_traj'], 0.3)
#     short = shorten_trajectory(new_traj)
#     if len(short) > 0:
#         action_freqs = make_action_frequency_vector(new_traj)
#         random_vecs_30.append({'sentence': i['sentence'], 'full_traj': new_traj, 'short_traj': short, 'action_freqs': action_freqs, 'label': 0})
        
# # random vecs with 1/2 noise / 0
# random_vecs_50 = []
# for i in full_data:
#     new_traj = add_noise(full_data[random.randint(0, len(full_data) - 1)]['full_traj'], 0.5)
#     short = shorten_trajectory(new_traj)
#     if len(short) > 0:
#         action_freqs = make_action_frequency_vector(new_traj)
#         random_vecs_50.append({'sentence': i['sentence'], 'full_traj': new_traj, 'short_traj': short, 'action_freqs': action_freqs, 'label': 0})

# # random vecs with 3/4 noise / 0
# random_vecs_75 = []
# for i in full_data:
#     new_traj = add_noise(full_data[random.randint(0, len(full_data) - 1)]['full_traj'], 0.75)
#     short = shorten_trajectory(new_traj)
#     if len(short) > 0:
#         action_freqs = make_action_frequency_vector(new_traj)
#         random_vecs_75.append({'sentence': i['sentence'], 'full_traj': new_traj, 'short_traj': short, 'action_freqs': action_freqs, 'label': 0})


# # fully random vecs / 0
# fully_random_vecs = []
# for i in full_data:
#     new_traj = [random.randint(1, 17) for x in range(len(i['full_traj']))]
#     short = shorten_trajectory(new_traj)
#     if len(short) > 0:
#         action_freqs = make_action_frequency_vector(new_traj)
#         fully_random_vecs.append({'sentence': i['sentence'], 'full_traj': new_traj, 'short_traj': short, 'action_freqs': action_freqs, 'label': 0})
        
#     print(full_data[random.randint(0, len(full_data))])