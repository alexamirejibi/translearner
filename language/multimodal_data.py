import pickle
from transformers import DistilBertForSequenceClassification, DistilBertConfig, Trainer, TrainingArguments, DistilBertTokenizerFast, DataCollatorWithPadding
import random
from sklearn.model_selection import train_test_split
import sys
from dataset import AnnotatedTrajectoryDataset
from datasets import load_metric
import numpy as np
from huggingface_hub import notebook_login
from shorten_trajectory import *


full_sent_traj_pairs = []
short_sent_traj_pairs = []
full_data = []

with open('data/full_sent_traj_pairs.pkl', 'rb') as f:
    data = pickle.load(f)

k = 0
for i in data:
    full = [x for x in i['trajectory'] if x != 0]
    short = shorten_trajectory(full)
    id = k
    if len(short) > 0:
        full_data.append({'id': id, 'sentence': i['sentence'], 'full': full, 'short': short})
        k += 1


# with open('data/short_traj_pairs_5.pkl', 'rb') as f:
#     data = pickle.load(f)
#     for i in data:
#         short_sent_traj_pairs.append({'sentence': i['sentence'],
#         'trajectory': i['trajectory']})

# for i in range(len(full_sent_traj_pairs)):
#     full_data.append({'sentence': full_sent_traj_pairs[i]['sentence'],
#     'full': full_sent_traj_pairs[i]['trajectory'], 'short': short_sent_traj_pairs[i]['trajectory']})

print(len(full_data))
print()