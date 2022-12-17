from transformers import BertConfig, pipeline, Trainer, TrainingArguments
import pandas as pd
import numpy as np
from utils import *

from scipy.special import softmax

from transformers import EvalPrediction
import torch 
from multimodal_transformers.model import BertWithTabular, AutoModelWithTabular, DistilBertWithTabular
from multimodal_transformers.model import TabularConfig
from multimodal_transformers.data import load_data_from_folder, load_data
#from multimodal_model import ModelArguments, MultimodalDataTrainingArguments
from dclasses import ModelArguments, MultimodalDataTrainingArguments
from transformers import AutoTokenizer, AutoConfig

from sklearn.preprocessing import PowerTransformer, QuantileTransformer




text_cols = ['description', 'short_traj']
cat_cols = []
from utils import action_words
numerical_cols = action_words

column_info_dict = {
    'text_cols': text_cols,
    'num_cols': numerical_cols,
    'label_col': 'label',
    'label_list': ['0', '1', '2', '3', '4']
}

data_args = MultimodalDataTrainingArguments(
    data_path='data/new_split/',
    combine_feat_method='gating_on_cat_and_num_feats_then_sum',
    column_info=column_info_dict,
    task='classification',
    categorical_encode_type='none'
)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

train_dataset, val_dataset, test_dataset = load_data_from_folder(
    data_args.data_path,
    data_args.column_info['text_cols'],
    tokenizer,
    label_col=data_args.column_info['label_col'],
    label_list=data_args.column_info['label_list'],
    categorical_encode_type='none',
    # categorical_cols=data_args.column_info['cat_cols'],
    numerical_cols=data_args.column_info['num_cols'],
    sep_text_token_str=tokenizer.sep_token,
    debug=True,
    numerical_transformer_method='none'
)

num_labels = len(np.unique(train_dataset.labels))
config = AutoConfig.from_pretrained('alexamiredjibi/multimodal-traj-class-no-numtransform')
tabular_config = TabularConfig(num_labels=num_labels,
                               # cat_feat_dim=train_dataset.cat_feats.shape[1],
                               numerical_feat_dim=train_dataset.numerical_feats.shape[1],
                               **vars(data_args))

config.tabular_config = tabular_config

model = AutoModelWithTabular.from_pretrained(
        'alexamiredjibi/multimodal-traj-class-no-numtransform',
        config=config,
        # cache_dir=model_args.cache_dir
    )

trainer = Trainer(
    model=model,
    compute_metrics=calc_classification_metrics,
)


# ------------------ ------------------

testset = pd.read_csv('data/new_split/test.csv')

# for i in range(5):
# tor_data = load_data(testset,
#                     text_cols=text_cols,
#                     tokenizer=tokenizer,
#                     label_col='label',
#                     label_list=['0', '1', '2', '3', '4'],
#                     categorical_encode_type='none',
#                     numerical_cols=numerical_cols,
#                     sep_text_token_str=tokenizer.sep_token,
#                     numerical_transformer='none',
#                     debug=True)

# print(tor_data[0].keys())
# print(tor_data.__len__())

print(' ------------------------------------')
# tor_data = load_data(testset,
#                     text_cols=text_cols,
#                     tokenizer=tokenizer,
#                     label_col='label',
#                     label_list=['0', '1', '2', '3', '4'],
#                     categorical_encode_type='none',
#                     numerical_cols=numerical_cols,
#                     sep_text_token_str=tokenizer.sep_token,
#                     debug=True)

# print(tor_data[0]['input_ids'].shape)

# predictions = trainer.predict(tor_data)[0]

# print(predictions)

# print(trainer.evaluate(test_dataset))
# print(trainer.evaluate(test_dataset))

# print('-----', np.argmax(predictions[0]))

# print(testset[:10]['label'])
# print(predictions.predictions[0])


def get_prediction(one_data):
    prediction = trainer.predict(one_data)[0][0]
    print("----------------------------------------")
    print(np.argmax(prediction))
    print(softmax(prediction))
    
descriptions=['go left and jump over the skull',
                'go left and jump left over the skull to reach the ladder',
                'goes left, jumps over the skull, goes left']
i = 4
action_freqs = np.array([0., 0.05, 0.04, 0.04, 0.07, 0.05, 0.05, 0.09, 0.11, 0.09, 0.05, 0.04, 0.04, 0.05,
  0.07, 0.04, 0.11, 0.04])
d = get_torch_data(descriptions, "LEFT, RIGHT, JUMP RIGHT, LEFT, JUMP RIGHT, LEFT, LEFT, LEFT, JUMP RIGHT, JUMP RIGHT, RIGHT, LEFT, LEFT, LEFT, LEFT", tokenizer=tokenizer, action_freqs=action_freqs)#trajectory=np.array([4, 4, 4, 4, 4, 4, 13, 4, 4, 4, 4, 4]))
# get_prediction(d)
# preds = trainer.predict(d)[0][0]
# preds = softmax(preds[0])
# print(np.round(preds, 3))
# print(np.argmax(preds))
# preds = np.average(preds, axis=0)
# preds = np.round(preds, 4)
# score = 0.1 * preds[1] + 0.2 * preds[2] + 0.3 * preds[3] + 0.4 * preds[4]
# print(score)

# print(test_dataset[:500].keys())
pred_labels = np.array([])
ac_labels = np.array([])
print(testset.keys())
print(testset.iloc[0].to_numpy()[-18:])

acc = 0
relacc = 0
n = 100
label1_acc = 0
total1 = 0

""" This is a little test to see if my data loading method is working correctly.
It passes on the test set with 75% accuracy, and 87% binary accuracy in guessing
whether data is positive or negative (1,2,3,4 or 0).

And yet the predictions are still not good in practice. I am fairly certain that my
implementation of the gym wrapper is correct. I also think the training data is fairly
good for the task. I am not sure what is going wrong.
"""

for i in range(n):
    label = testset['label'].iloc[i]
    ac_labels = np.append(ac_labels, label)
    # print(i.values)
    short_traj = testset['short_traj'].iloc[i]
    description = testset['description'].iloc[i]
    action_freqs = testset.iloc[i].to_numpy()[-18:]
    action_freqs = action_freqs.astype(float)
    d = get_torch_data([description, description, description], short_traj, tokenizer=tokenizer, action_freqs=action_freqs)
    preds = trainer.predict(d)[0][0]
    print(description, short_traj, action_freqs, label)
    preds = softmax(preds[0])
    pred_label = np.argmax(preds)
    print('preds', preds, 'label', pred_label)
    acc = acc + (pred_label == label)
    relacc = relacc + (1 if (pred_label == label or (pred_label > 0 and label > 0))
                       else 0)
    total1 = total1 + (1 if label == 1 else 0)
    label1_acc = label1_acc + (1 if (pred_label <= 1 and label == 1) else 0)
    # pred_labels = np.append(pred_labels, pred_label)

print('acc', acc / n)
print('relacc', relacc / n)
print('label1_acc', label1_acc / total1)
print('TEST SUCCESSFUL' if acc / n > 0.70 else 'TEST FAILED', 'acc', acc / n)
# acc = 0
# for i in range(100):
#     print(pred_labels[i], ac_labels[i])
#     acc = acc + (pred_labels[i] == ac_labels[i])
    
# print(acc/100)

# ev = EvalPrediction(predictions=pred_labels, label_ids=ac_labels)
# print(calc_classification_metrics(ev))

print(' ------------------ ------------------')
        