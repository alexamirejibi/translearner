from dataclasses import dataclass, field
import json
import logging
import os
from typing import Optional
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
os.environ['COMET_MODE'] = 'DISABLED'

data_df = pd.read_csv('data/augmented_data.csv')

print(data_df.describe(include=np.object))

train_df, val_df, test_df = np.split(data_df.sample(frac=1), [int(.8*len(data_df)), int(.9 * len(data_df))])
print('Num examples train-val-test')
print(len(train_df), len(val_df), len(test_df))


train_df.to_csv('data/new_split/train.csv')
val_df.to_csv('data/new_split/val.csv')
test_df.to_csv('data/new_split/test.csv')