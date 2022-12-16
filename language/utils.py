import numpy as np
from numpy import ndarray
from scipy.special import softmax
from sklearn.metrics import matthews_corrcoef

from dclasses import ModelArguments, MultimodalDataTrainingArguments

from transformers import EvalPrediction

from multimodal_transformers.data.tabular_torch_dataset import TorchTabularTextDataset

from sklearn.preprocessing import QuantileTransformer, PowerTransformer



ACTION_WORDS = ['STAND', 'JUMP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UP-RIGHT', 'UP-LEFT', 'DOWN-RIGHT', 'DOWN-LEFT', 'JUMP UP', 'JUMP RIGHT', 'JUMP LEFT', 'JUMP DOWN', 'JUMP UP-RIGHT', 'JUMP UP-LEFT', 'JUMP DOWN-RIGHT', 'JUMP DOWN-LEFT']
SIMPLIFIED_ACTIONS = ['STAND', 'JUMP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'RIGHT', 'LEFT', 'RIGHT', 'LEFT', 'JUMP', 'JUMP RIGHT', 'JUMP LEFT', 'JUMP DOWN', 'JUMP RIGHT', 'JUMP LEFT', 'JUMP RIGHT', 'JUMP LEFT']

SHORT_TRAJ_LEN = 15

action_words = ACTION_WORDS # used these already

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
    combine_feat_method='gating_on_cat_and_num_feats_then_sum',
    column_info=column_info_dict,
    task='classification',
    categorical_encode_type='none'
)


def make_action_frequency_vector(trajectory):
    # count occurences of every element in trajectory
    if isinstance(trajectory, list):
        trajectory = np.array(trajectory)
    l = len(trajectory)
    frequencies = np.array([round(np.count_nonzero(trajectory == i) / l, 2) for i in range(18)])
    return frequencies


def traj_to_words(trajectory, simple=False):
    """Convert trajectory to words
    Args:
        trajectory: Trajectory
    Returns:
        _type_: List of words
    """
    if simple:
        words = [SIMPLIFIED_ACTIONS[x] for x in trajectory]
    else:
        words = [ACTION_WORDS[x] for x in trajectory]
    return words


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
    num_fours = np.count_nonzero(pred_labels == 4)
    relacc = [1 if 
              # both are the same or both are non-zero
              (pred_labels[i] == labels[i] 
               or (pred_labels[i] > 0 and labels[i] > 0)) 
              else 0 for i in range(len(pred_labels))]
    relacc = np.mean(relacc)
    result = {
        "acc": acc,
        'relacc': relacc,
        "num_fours": num_fours,
        "mcc": matthews_corrcoef(labels, pred_labels)
        }

    return result


numerical_transformer = QuantileTransformer(output_distribution='normal')

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
    
    # transformer = numerical_transformer.fit(action_freqs)
    # action_freqs = transformer.transform(action_freqs)
    # action_freqs = action_freqs.astype(float)
    
    description = [description, description]
    short_traj = [short_traj, short_traj]
    
    hf_model_text_input = tokenizer(description, short_traj, padding=True, truncation=True)
    labels = np.array([1, 1])

    return TorchTabularTextDataset(hf_model_text_input,
                                   numerical_feats=action_freqs,
                                   label_list=label_list, categorical_feats=None, labels=labels)