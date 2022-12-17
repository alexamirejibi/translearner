from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    set_seed
)
from transformers.training_args import TrainingArguments

from multimodal_transformers.data import load_data_from_folder
from multimodal_transformers.model import TabularConfig
from multimodal_transformers.model import AutoModelWithTabular

from utils import *

from dclasses import ModelArguments, MultimodalDataTrainingArguments

train_df = pd.read_csv('data/new_split/train.csv')
val_df = pd.read_csv('data/new_split/val.csv')
test_df = pd.read_csv('data/new_split/test.csv')

text_cols = ['description', 'short_traj']
cat_cols = []

numerical_cols = ACTION_WORDS
print(numerical_cols)

model_args = ModelArguments(
    model_name_or_path='distilbert-base-uncased'
)

tokenizer_path_or_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
print('Specified tokenizer: ', tokenizer_path_or_name)
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path_or_name,
    cache_dir=model_args.cache_dir,
)


# load data from data/new_split
train_dataset, val_dataset, test_dataset = load_data_from_folder(
    data_args.data_path,
    data_args.column_info['text_cols'],
    tokenizer,
    label_col=data_args.column_info['label_col'],
    label_list=data_args.column_info['label_list'],
    categorical_encode_type='none',
    numerical_cols=data_args.column_info['num_cols'],
    sep_text_token_str=tokenizer.sep_token,
    numerical_transformer_method='none',
)


# configs for training
num_labels = len(np.unique(train_dataset.labels))
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


training_args = TrainingArguments(
    output_dir="multimodal-traj-class-no-numtransform",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=20,
    eval_steps=250,
    load_best_model_at_end=True,
    metric_for_best_model='relacc',
    save_strategy='epoch',
    evaluation_strategy='epoch',
    push_to_hub=True,
)

set_seed(training_args.seed)

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