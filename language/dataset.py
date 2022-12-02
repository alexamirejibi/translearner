import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from torch.optim import Adam
from tqdm import tqdm

from sklearn.model_selection import train_test_split

# https://huggingface.co/transformers/v3.2.0/custom_datasets.html
# LEFT OFF TODO: I was on this page, trying to set up this custom dataset class.
# I need to pass the data as encodings. the definitions below need to be fixed.

class AnnotatedTrajectoryDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        # self.sentences = [i['sentence'] for i in sentence_trajectory_pairs]
        # self.trajectories = [i['trajectory'] for i in sentence_trajectory_pairs]
        # # self.clip_ids = [i['clip_id'] for i in sentence_trajectory_pairs]
        # self.labels = [i['label'] for i in sentence_trajectory_pairs]
        # self.sentences = [tokenizer(sentence, 
        #                        padding='max_length', max_length = 512, truncation=True,
        #                         return_tensors="pt") for sentence in df['sentence']]
        # self.trajectories = [tokenizer(trajectory, 
        #                        padding='max_length', max_length = 512, truncation=True,
        #                         return_tensors="pt") for trajectory in df['trajectory']]
        for l in self.labels:
            assert l in [0, 1]
    


    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    # def get_batch_labels(self, idx):
    #     # Fetch a batch of labels
    #     return np.array(self.labels[idx])
    
    # def get_batch_pairs(self, idx):
    #     # Fetch a batch of inputs
    #     return self.encodings[idx]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item