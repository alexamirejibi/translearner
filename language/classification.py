from transformers import BertConfig, pipeline, Trainer, TrainingArguments
import pandas as pd
import numpy as np
from utils import calc_classification_metrics
import torch 
from multimodal_transformers.model import BertWithTabular, AutoModelWithTabular, DistilBertWithTabular
from multimodal_transformers.model import TabularConfig
from multimodal_transformers.data import load_data_from_folder
#from multimodal_model import ModelArguments, MultimodalDataTrainingArguments
from dclasses import ModelArguments, MultimodalDataTrainingArguments
from transformers import AutoTokenizer, AutoConfig


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

data_args = MultimodalDataTrainingArguments(
    data_path='data/new_split/',
    combine_feat_method='individual_mlps_on_cat_and_numerical_feats_then_concat',
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
)

num_labels = len(np.unique(train_dataset.labels))
config = AutoConfig.from_pretrained('alexamiredjibi/Multimodal-Trajectory-Classifier')
tabular_config = TabularConfig(num_labels=num_labels,
                               # cat_feat_dim=train_dataset.cat_feats.shape[1],
                               numerical_feat_dim=train_dataset.numerical_feats.shape[1],
                               **vars(data_args))

config.tabular_config = tabular_config

model = DistilBertWithTabular.from_pretrained(
        'alexamiredjibi/Multimodal-Trajectory-Classifier',
        config=config,
        # cache_dir=model_args.cache_dir
    )

training_args = TrainingArguments(
    output_dir="Multimodal-Trajectory-Classifier",
    #logging_dir="Multimodal-Trajectory-Classifier/logs/runs",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=32,
    num_train_epochs=1,
    # evaluate_during_training=True,
    logging_steps=25,
    eval_steps=250,
    use_mps_device=True, # TODO check if this is needed  <<<<<<--------------------------<<<<<<
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=calc_classification_metrics,
)

print(trainer.evaluate())

#preds, label_ids, metrics = trainer.predict(test_dataset)

#print(preds)
#print(trainer.evaluate())

# save predictions
#np.save('preds.npy', preds)

# load predictions
np.load('preds.npy', allow_pickle=True)

# # preds vs label_ids
# correct = 0
# total = 0
# for i in range(len(preds)):
#     if preds[i] == label_ids[i]:
#         #print("Correct")
#         correct += 1
#     total += 1

# print("Accuracy: ", correct/total)
        