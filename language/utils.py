# import numpy as np
# import torch
# import torch.nn as nn

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
)

from transformers import EvalPrediction

action_words = ['STAND', 'JUMP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UP-RIGHT', 'UP-LEFT', 'DOWN-RIGHT', 'DOWN-LEFT', 'JUMP UP', 'JUMP RIGHT', 'JUMP LEFT', 'JUMP DOWN', 'JUMP UP-RIGHT', 'JUMP UP-LEFT', 'JUMP DOWN-RIGHT', 'JUMP DOWN-LEFT']
short_traj_len = 10

def calc_classification_metrics(p: EvalPrediction):
    pred_labels = np.argmax(p.predictions[0], axis=1)
    pred_scores = softmax(p.predictions[0], axis=1)
    labels = p.label_ids
    if len(np.unique(labels)) == 2:  # binary classification
        roc_auc_pred_score = roc_auc_score(labels, pred_scores)
        precisions, recalls, thresholds = precision_recall_curve(labels,
                                                                    pred_scores)
        fscore = (2 * precisions * recalls) / (precisions + recalls)
        fscore[np.isnan(fscore)] = 0
        ix = np.argmax(fscore)
        threshold = thresholds[ix].item()
        pr_auc = auc(recalls, precisions)
        tn, fp, fn, tp = confusion_matrix(labels, pred_labels, labels=[0, 1]).ravel()
        result = {'roc_auc': roc_auc_pred_score,
                    'threshold': threshold,
                    'pr_auc': pr_auc,
                    'recall': recalls[ix].item(),
                    'precision': precisions[ix].item(), 'f1': fscore[ix].item(),
                    'tn': tn.item(), 'fp': fp.item(), 'fn': fn.item(), 'tp': tp.item()
                    }
    else:
        acc = (pred_labels == labels).mean()
        relacc = [1 if (pred_labels[i] == labels[i] or (pred_labels[i] > 0 and labels[i] > 0)) else 0 for i in range(len(pred_labels))]
        relacc = np.mean(relacc)
        result = {
            "acc": acc,
            'relacc': relacc,
            "mcc": matthews_corrcoef(labels, pred_labels)
        }

    return result
# # environment parameters
# ENV_NAME = 'MontezumaRevenge-v0'
# SCREEN_WIDTH = 84
# SCREEN_HEIGHT = 84
# N_ACTIONS = 18
# MAX_STEPS = 1000
# RANDOM_START = 30

# # PPO parameters
# value_loss_coef = 0.5
# entropy_coef = 0.01
# lr = 7e-4
# eps = 1e-5
# alpha = 0.99
# max_grad_norm = 0.5
# clip_param = 0.2
# ppo_epoch = 4
# num_mini_batch = 8
# num_processes = 1
# num_steps = 64
# num_stack = 1
# use_gae = False
# gamma = 0.99
# tau = 0.95
# log_interval = 100
# obs_shape = (num_stack, 84, 84)
# num_updates = 1000000


# device = 'cuda'

# spearman_corr_coeff_actions = [0, 1, 2, 3, 4, 5, 11, 12]

# def rgb2gray(image):
#   return np.dot(image[...,:3], [0.299, 0.587, 0.114])


# # Necessary for my KFAC implementation.
# class AddBias(nn.Module):
#     def __init__(self, bias):
#         super(AddBias, self).__init__()
#         self._bias = nn.Parameter(bias.unsqueeze(1))

#     def forward(self, x):
#         if x.dim() == 2:
#             bias = self._bias.t().view(1, -1)
#         else:
#             bias = self._bias.t().view(1, -1, 1, 1)

#         return x + bias


# def init(module, weight_init, bias_init, gain=1):
#     weight_init(module.weight.data, gain=gain)
#     bias_init(module.bias.data)
#     return module


# # https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
# def init_normc_(weight, gain=1):
#     weight.normal_(0, 1)
#     weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


# def update_current_obs(obs, current_obs, obs_shape, num_stack):
#     shape_dim0 = obs_shape[0]
#     obs = torch.from_numpy(obs).float()
#     if num_stack > 1:
#         current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
#     current_obs[:, -shape_dim0:] = obs
