import pickle
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizerFast, DataCollatorWithPadding
import random
from sklearn.model_selection import train_test_split
from dataset import AnnotatedTrajectoryDataset
import evaluate
import numpy as np
import torch
from datasets import load_metric


sentence_trajectory_pairs = []

with open('data/sentence_action_pairs.pkl', 'rb') as f:
    data = pickle.load(f)
    for i in data:
        sentence_trajectory_pairs.append({'sentence': i['sentence'],
        'trajectory': i['trajectory']})
print(sentence_trajectory_pairs[0])

neg_examples = []
for i in range(len(sentence_trajectory_pairs)):
    random_trajectory = random.choice(sentence_trajectory_pairs)['trajectory']
    while random_trajectory == sentence_trajectory_pairs[i]['trajectory']:
        random_trajectory = random.choice(sentence_trajectory_pairs)['trajectory']
    neg_examples.append({'sentence': sentence_trajectory_pairs[i]['sentence'], 'trajectory': random_trajectory})

pos_labels = [1 for i in range(len(sentence_trajectory_pairs))]
neg_labels = [0 for i in range(len(neg_examples))]

all_pairs = sentence_trajectory_pairs + neg_examples
all_labels = pos_labels + neg_labels

assert len(all_labels) == len(all_pairs)

for l in all_labels:
    assert l in [0, 1]

train_dataset, test_dataset, train_labels, test_labels = train_test_split(all_pairs, all_labels, test_size=0.3, random_state=42)
val_dataset, test_dataset, val_labels, test_labels = train_test_split(test_dataset, test_labels, test_size=0.5, random_state=42)

train_sentences = [i['sentence'] for i in train_dataset]
train_trajectories = [i['trajectory'] for i in train_dataset]

val_sentences = [i['sentence'] for i in val_dataset]
val_trajectories = [i['trajectory'] for i in val_dataset]
test_sentences = [i['sentence'] for i in test_dataset]
test_trajectories = [i['trajectory'] for i in test_dataset]

assert len(train_sentences) == len(train_trajectories)
assert len(val_sentences) == len(val_trajectories)
assert len(test_sentences) == len(test_trajectories)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_sentences, train_trajectories, truncation=True, padding=True)
val_encodings = tokenizer(val_sentences, val_trajectories, truncation=True, padding=True)
test_encodings = tokenizer(test_sentences, test_trajectories, truncation=True, padding=True)

# print 10 random decoded train_encodings
# for i in range(20):
#     print(tokenizer.decode(train_encodings['input_ids'][i]))
#     print('\n')

train_dataset = AnnotatedTrajectoryDataset(train_encodings, train_labels)
val_dataset = AnnotatedTrajectoryDataset(val_encodings, val_labels)
test_dataset = AnnotatedTrajectoryDataset(test_encodings, test_labels)

# check that items in train_dataset are of type dict
# print(train_pairs[0])
# print(train_pairs.get_batch_labels(0))

# model = DistilBertForSequenceClassification.from_pretrained('model_learn_pretrained')
#tokenizer = DistilBertTokenizerFast.from_pretrained('results')
model = DistilBertForSequenceClassification.from_pretrained("model_learn_pretrained")

tst = tokenizer('go left and then jump', 'LEFT, JUMP', truncation=True, padding=True, return_tensors='pt')

with torch.no_grad():
    logits = model(**tst).logits

load_accuracy = load_metric("accuracy")
load_f1 = load_metric("f1")

logits, labels = eval_pred
predictions = np.argmax(logits, axis=-1)
accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]





