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


preds = np.load('preds.npy', allow_pickle=True)
# print(np.argmax(preds[0][0]))
print(preds[0][0].shape)
