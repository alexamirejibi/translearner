import preprocessing as pre
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import random
from shorten_trajectory import shorten_trajectory
import torch
import evaluate
import numpy as np

sentence_trajectory_pairs = []
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def load_sentence_trajectory_pairs(path='data/sentence_action_pairs.pkl'):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        for i in data:
            sentence_trajectory_pairs.append(i)

def preprocess_function(sent_traj_pairs):
    return tokenizer(sent_traj_pairs["text"], truncation=True)

def prepare_stuff():
    shuffled_data = sentence_trajectory_pairs
    random.shuffle(shuffled_data)
    # split
    training_size = int(len(shuffled_data) * 0.8)
    training_data = shuffled_data[:training_size]
    validation_data = shuffled_data[training_size:]

    train_descriptions = [training_data[i]['sentence'] for i in range(len(training_data))]
    train_real_trajectories = [training_data[i]['trajectory'] for i in range(len(training_data))]
    train_random_trajectories = [training_data[i]['trajectory'] for i in range(len(training_data))]
    random.shuffle(train_random_trajectories)

    valid_descriptions = [validation_data[i]['sentence'] for i in range(len(validation_data))]
    valid_real_trajectories = [validation_data[i]['trajectory'] for i in range(len(validation_data))]
    valid_random_trajectories = [validation_data[i]['trajectory'] for i in range(len(validation_data))]
    random.shuffle(valid_random_trajectories)
    # list of all trajectories

    classes = ["unrelated", "related"]
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")

    train_related_pairs = tokenizer(train_descriptions, train_real_trajectories, return_tensors="pt", padding="max_length", truncation=True)
    train_unrelated_pairs = tokenizer(train_descriptions, train_random_trajectories, return_tensors="pt", padding="max_length", truncation=True)
    train_labeled = train_related_pairs + train_unrelated_pairs

    valid_related_pairs = tokenizer(valid_descriptions, valid_real_trajectories, return_tensors="pt", padding="max_length", truncation=True)
    valid_unrelated_pairs = tokenizer(valid_descriptions, valid_random_trajectories, return_tensors="pt", padding="max_length", truncation=True)

    return train_related_pairs, train_unrelated_pairs, validation_data
    # related_classification_logits = model(**related).logits
    # not_related_classification_logits = model(**unrelated).logits

    # related_results = torch.softmax(related_classification_logits, dim=1).tolist()[0]
    # not_related_results = torch.softmax(not_related_classification_logits, dim=1).tolist()[0]

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# TRAINING --------------------------------------------------------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
def train():
    # train the model
    train_related_pairs, train_unrelated_pairs, validation_data = prepare_stuff(sentence_trajectory_pairs)
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
