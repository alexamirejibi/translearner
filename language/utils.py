import numpy as np
from numpy import ndarray
from scipy.special import softmax
from sklearn.metrics import matthews_corrcoef

from dclasses import ModelArguments, MultimodalDataTrainingArguments

from transformers import EvalPrediction

from multimodal_transformers.data.tabular_torch_dataset import TorchTabularTextDataset


ACTION_WORDS = ['STAND', 'JUMP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UP-RIGHT', 'UP-LEFT', 'DOWN-RIGHT', 'DOWN-LEFT', 'JUMP UP', 'JUMP RIGHT', 'JUMP LEFT', 'JUMP DOWN', 'JUMP UP-RIGHT', 'JUMP UP-LEFT', 'JUMP DOWN-RIGHT', 'JUMP DOWN-LEFT']
SHORT_TRAJ_LEN = 10

action_words = ACTION_WORDS # used these already
short_traj_len = SHORT_TRAJ_LEN

text_cols = ['description', 'short_traj']
label_list = ['0', '1', '2', '3', '4']
numerical_cols = ACTION_WORDS


column_info_dict = {
    'text_cols': text_cols,
    'num_cols': numerical_cols,
    'label_col': 'label',
    'label_list': label_list
}


data_args = MultimodalDataTrainingArguments(
    data_path='data/new_split/',
    combine_feat_method='individual_mlps_on_cat_and_numerical_feats_then_concat',
    column_info=column_info_dict,
    task='classification',
    categorical_encode_type='none'
)


def make_action_frequency_vector(trajectory:ndarray):
    # count occurences of 0 in trajectory
    frequencies = [round(np.count_nonzero(trajectory == x) / len(trajectory), ndigits=2) for x in range(18)]
    return frequencies


def calc_classification_metrics(p: EvalPrediction):
    """Calculate classification metrics for model predictions

    Args:
        p (EvalPrediction): Eval predictions
    Returns:
        _type_: Results
    """
    pred_labels = np.argmax(p.predictions[0], axis=1)
    pred_scores = softmax(p.predictions[0], axis=1)
    labels = p.label_ids
    acc = (pred_labels == labels).mean()
    relacc = [1 if 
              # both are the same or both are non-zero
              (pred_labels[i] == labels[i] 
               or (pred_labels[i] > 0 and labels[i] > 0)) 
              else 0 for i in range(len(pred_labels))]
    relacc = np.mean(relacc)
    result = {
        "acc": acc,
        'relacc': relacc,
        "mcc": matthews_corrcoef(labels, pred_labels)
        }

    return result


def get_torch_data(description,
                  short_traj,
                  tokenizer,
                  trajectory=None,
                  action_freqs=None,
                  ):
    """Function to load a single dataset given a trajectory
    Args:
        tokenizer (:obj:`transformers.tokenization_utils.PreTrainedTokenizer`):
            HuggingFace tokenizer used to tokenize the input texts as specifed by text_cols
    Returns:
        :obj:`tabular_torch_dataset.TorchTextDataset`: The converted dataset
    """
    if action_freqs is None:
        if trajectory is None:
            raise ValueError('Either trajectory or action_freqs must be provided')
        action_freqs = make_action_frequency_vector(trajectory)
    
    action_freqs = np.array([action_freqs, action_freqs])
    # action_freqs = action_freqs.astype(float)
    
    description = [description, description]
    short_traj = [short_traj, short_traj]
    
    hf_model_text_input = tokenizer(description, short_traj, padding=True, truncation=True)
    labels = np.array([1, 1])

    return TorchTabularTextDataset(hf_model_text_input,
                                   numerical_feats=action_freqs,
                                   label_list=label_list, categorical_feats=None, labels=labels)