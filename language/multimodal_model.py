from dataclasses import dataclass, field
import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    EvalPrediction,
    set_seed
)
from transformers.training_args import TrainingArguments

from multimodal_transformers.data import load_data_from_folder
from multimodal_transformers.model import TabularConfig
from multimodal_transformers.model import AutoModelWithTabular

from utils import calc_classification_metrics

from dclasses import ModelArguments, MultimodalDataTrainingArguments

train_df = pd.read_csv('data/new_split/train.csv')
val_df = pd.read_csv('data/new_split/val.csv')
test_df = pd.read_csv('data/new_split/test.csv')

text_cols = ['description', 'short_traj']
cat_cols = []
from utils import action_words
numerical_cols = action_words
print(numerical_cols)

column_info_dict = {
    'text_cols': text_cols,
    'num_cols': numerical_cols,
    #'cat_cols': cat_cols,
    'label_col': 'label',
    'label_list': ['0', '1', '2', '3', '4']
}


model_args = ModelArguments(
    model_name_or_path='distilbert-base-uncased'
)

data_args = MultimodalDataTrainingArguments(
    data_path='data/new_split/',
    combine_feat_method='individual_mlps_on_cat_and_numerical_feats_then_concat',
    column_info=column_info_dict,
    task='classification',
    categorical_encode_type='none'
)

training_args = TrainingArguments(
    output_dir="Multimodal-Trajectory-Classifier",
    #logging_dir="Multimodal-Trajectory-Classifier/logs/runs",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=32,
    num_train_epochs=10,
    # evaluate_during_training=True,
    logging_steps=25,
    eval_steps=250,
    use_mps_device=True, # TODO check if this is needed  <<<<<<--------------------------<<<<<<
    push_to_hub=True,
)

set_seed(training_args.seed)

tokenizer_path_or_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
print('Specified tokenizer: ', tokenizer_path_or_name)
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path_or_name,
    cache_dir=model_args.cache_dir,
)

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
)

num_labels = len(np.unique(train_dataset.labels))
print(num_labels)

config = AutoConfig.from_pretrained('distilbert-base-uncased')
tabular_config = TabularConfig(num_labels=num_labels,
                               # cat_feat_dim=train_dataset.cat_feats.shape[1],
                               numerical_feat_dim=train_dataset.numerical_feats.shape[1],
                               **vars(data_args))
config.tabular_config = tabular_config

model = AutoModelWithTabular.from_pretrained(
        'distilbert-base-uncased',
        config=config,
        cache_dir=model_args.cache_dir
    )


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=calc_classification_metrics,
)

trainer.train()
trainer.push_to_hub()
print("eval", trainer.evaluate(), " test: ", trainer.evaluate(test_dataset))