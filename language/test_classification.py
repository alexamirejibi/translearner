from transformers import BertConfig, pipeline, Trainer, TrainingArguments
import pandas as pd
import numpy as np
from utils import *

from scipy.special import softmax


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
from utils import ACTION_WORDS
numerical_cols = ACTION_WORDS

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
preds = trainer.predict(d)[0][0]
preds = softmax(preds[0])
print(np.round(preds, 3))
print(np.argmax(preds))
# preds = np.average(preds, axis=0)
# preds = np.round(preds, 4)
# score = 0.1 * preds[1] + 0.2 * preds[2] + 0.3 * preds[3] + 0.4 * preds[4]
# print(score)


print(' ------------------ ------------------')
# print(trainer.evaluate(tor_data))
# d = get_torch_data('go right and down the ladder', "LEFT, UP, LEFT, JUMP-LEFT, UP, DOWN, LEFT, RIGHT, UP", tokenizer=tokenizer, trajectory=np.array([4, 4, 4, 4, 4, 4, 13, 4, 4, 4, 4, 4]))

# with torch.no_grad():
#     preds = trainer.predict(d).predictions[0][0]
#     print(softmax(preds), np.argmax(preds))

# print(np.argmax(preds[0]))
# print(softmax(preds[0]))



# with torch.no_grad():
#     _, logits, classifier_outputs = model(
#         model_inputs['input_ids'],
#         attention_mask=model_inputs['attention_mask'],
#         cat_feats=None,
#         #token_type_ids = model_inputs['token_type_ids'],
#         numerical_feats=model_inputs['numerical_feats']
#     )


#print(trainer.evaluate())

# model_inputs = test_dataset[0]
# print(model_inputs.keys())
# print(model_inputs['numerical_feats'].size())
# print(model_inputs)
# print(model)
# with torch.no_grad():
#     _, logits, classifier_outputs = model(
#         model_inputs['input_ids'],
#         attention_mask=model_inputs['attention_mask'],
#         cat_feats=None,
#         #token_type_ids = model_inputs['token_type_ids'],
#         numerical_feats=model_inputs['numerical_feats']
#     )

# print(classifier_outputs)


#preds, label_ids, metrics = trainer.predict(test_dataset)

#print(preds)
#print(trainer.evaluate())

# save predictions
#np.save('preds.npy', preds)

# load predictions
# np.load('preds.npy', allow_pickle=True)

# # preds vs label_ids
# correct = 0
# total = 0
# for i in range(len(preds)):
#     if preds[i] == label_ids[i]:
#         #print("Correct")
#         correct += 1
#     total += 1

# print("Accuracy: ", correct/total)
        