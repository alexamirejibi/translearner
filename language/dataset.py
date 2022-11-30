import torch
import numpy as np
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

from torch.optim import Adam
from tqdm import tqdm

# https://huggingface.co/transformers/v3.2.0/custom_datasets.html
# LEFT OFF TODO: I was on this page, trying to set up this custom dataset class.
# I need to pass the data as encodings. the definitions below need to be fixed.

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
labels = {'unrelated':0,
          'related':1,
          }

class AnnotatedTrajectoryDataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = [labels[label] for label in df['label']]
        self.enc_sentences = [tokenizer(sentence, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for sentence in df['sentence']]
        self.enc_trajectories = [tokenizer(trajectory, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for trajectory in df['trajectory']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_sentences(self, idx):
        # Fetch a batch of inputs
        return self.enc_sentences[idx]
    
    def get_batch_trajectories(self, idx):
        # Fetch a batch of inputs
        return self.enc_trajectories[idx]

    def __getitem__(self, idx):

        batch_sentences = self.get_batch_sentences(idx)
        batch_trajectories = self.get_batch_trajectories(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_sentences, batch_trajectories, batch_y