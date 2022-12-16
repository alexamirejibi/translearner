

from utils import SHORT_TRAJ_LEN, make_action_frequency_vector
import numpy as np
import random

from collections import Counter


# def make_action_frequency_vector(trajectory):
#     frequencies = [round(trajectory.count(x) / len(trajectory), ndigits=2) for x in range(18)]
#     return frequencies


def resize(traj, scale_to_n=SHORT_TRAJ_LEN):
    trajectory = traj.copy()
    # print('-------------------------')
    # print('original:', trajectory)
    
    if isinstance(trajectory, np.ndarray):
        trajectory = trajectory.tolist()
    
    trajectory = [i for i in trajectory if i != 0]
    
    if len(trajectory) == 0:
        return trajectory

    # if len(trajectory) > scale_to_n:
    #     trajectory = proportional_shorten(trajectory)
        
    if len(trajectory) > scale_to_n:
        trajectory = keep_common_actions(trajectory, SHORT_TRAJ_LEN + 10)
    
    t = randomly_resize_to_scale(trajectory, scale_to_n)
    # print('final:', t)
    # print('-------------------------')
    return t


max_traj_length = SHORT_TRAJ_LEN
    
def proportional_shorten(trajectory, scale_to_n=SHORT_TRAJ_LEN):
    shrink_factor = scale_to_n / len(trajectory)

    split_sequences = []
    tmp = []
    for i in range(len(trajectory)):
        if i == 0:
            tmp.append(trajectory[i])
        elif trajectory[i] == trajectory[i-1]:
            tmp.append(trajectory[i])
        elif len(tmp) == 0:
            tmp.append(trajectory[i])
        else:
            split_sequences.append(tmp)
            tmp = [trajectory[i]]
    split_sequences.append(tmp)
    

    new_traj = []
    for l in split_sequences:
        new_count = len(l) * shrink_factor
        new_traj.extend(l[:round(new_count)])
            
    return new_traj
    

def keep_common_actions(trajectory, scale_to_n=SHORT_TRAJ_LEN):
    # # print('here,', trajectory, scale_to_n)
    downsampled_actions = trajectory
    
    n = len(set(trajectory)) - 1
    # # print(n)
    while len(trajectory) > scale_to_n:
        if n <= 2:
            return randomly_resize_to_scale(trajectory, scale_to_n)

        action_counts = Counter(trajectory)
        most_common_actions = action_counts.most_common(n)
        most_common_actions = [action for action, count in most_common_actions]
        downsampled_actions = []
        for action in trajectory:
            if action in most_common_actions:
                downsampled_actions.append(action)
        
        trajectory = downsampled_actions
        # # print(n, trajectory)
        n = n - 1
        
    return downsampled_actions

def randomly_resize_to_scale(trajectory, scale_to_n=SHORT_TRAJ_LEN):

    while len(trajectory) > scale_to_n:
        trajectory.pop(random.randint(0, len(trajectory)-1))
    
    while len(trajectory) < scale_to_n:
        # print('SHORTER')
        # print(len(trajectory))
        random_ind = random.randint(0, len(trajectory)-1)
        trajectory.insert(random_ind, trajectory[random_ind])
        
    return trajectory


def divide_and_shorten(trajectory, splits=2, sofar=0):
    # # print(trajectory)
    if splits == 0:
        tmp = keep_common_actions(trajectory, scale_to_n=int(SHORT_TRAJ_LEN/(2 * sofar + 1)))
        # # print('final ', tmp)
        return tmp
    
    if splits > 0:
        m = int(len(trajectory) / 2)
        return divide_and_shorten(trajectory[:m], splits-1, sofar+1) + divide_and_shorten(trajectory[m:], splits-1, sofar+1)
    
    
    
# ---------------------------------------------

def shorten_traj_recency(trajectory):
    max_traj_length = 10
    length = len(trajectory)

    trajectory = [i for i in trajectory if i != 0]
    if len(trajectory) <= max_traj_length:
        return trajectory

    traj_len = len(trajectory)
    # # print(trajectory)


    split_sequences = []
    tmp = []
    for i in range(len(trajectory)):
        if i == 0:
            tmp.append(trajectory[i])
        elif trajectory[i] == trajectory[i-1]:
            tmp.append(trajectory[i])
        elif len(tmp) == 0:
            tmp.append(trajectory[i])
        else:
            split_sequences.append(tmp)
            tmp = [trajectory[i]]
    split_sequences.append(tmp)
    # # print(split_sequences)

    significance_coefficients = [] # the more recent the action, the more significant it is
    
    new_traj = []

    for i in range(len(split_sequences)):
        significance_coefficients.append(max(0.3, 1 - (1 / (i+1))))

    new_split_sequences = split_sequences.copy()
    ## print(len(new_split_sequences), len(significance_coefficients))
    while find_total_length(new_split_sequences) > max_traj_length:
        length = find_total_length(new_split_sequences)
        # # print(length)
        for i in range(len(new_split_sequences)):
            new_count = round(len(new_split_sequences[i]) * significance_coefficients[i])
            # if new_count == 0:
            #     new_count = 1
            new_split_sequences[i] = new_split_sequences[i][:new_count]
        if find_total_length(new_split_sequences) == length and length > max_traj_length:
            shrink_factor = max_traj_length / find_total_length(new_split_sequences)
            for i in range(len(new_split_sequences)):
                new_count = round(len(new_split_sequences[i]) * shrink_factor)
                # if new_count == 0:
                #     new_count = 1
                new_split_sequences[i] = new_split_sequences[i][:new_count]
            continue

    for l in new_split_sequences:
        new_traj.extend(l)

    return new_traj


def expand_trajectory(trajectory):
    split_sequences = []
    tmp = []
    for i in range(len(trajectory)):
        if i == 0:
            tmp.append(trajectory[i])
        elif trajectory[i] == trajectory[i-1]:
            tmp.append(trajectory[i])
        elif len(tmp) == 0:
            tmp.append(trajectory[i])
        else:
            split_sequences.append(tmp)
            tmp = [trajectory[i]]
    split_sequences.append(tmp)

    significance_coefficients = [] # the more recent the action, the more significant it is
    
    new_traj = []

    for i in range(len(split_sequences)):
        significance_coefficients.insert(0, 1 + (max(0.3, (1 / (i+1)))))

    new_split_sequences = split_sequences.copy()
    ## print(len(new_split_sequences), len(significance_coefficients))
    while find_total_length(new_split_sequences) < max_traj_length:
        #length = find_total_length(new_split_sequences)
        ## print('yes')
        for i in range(len(new_split_sequences)):
            new_count = round(len(new_split_sequences[i]) * significance_coefficients[i])
            new_split_sequences[i] = [] + [new_split_sequences[i][0]] * new_count

    for l in new_split_sequences:
        new_traj.extend(l)

    return new_traj


def find_total_length(split_sequences):
    return sum(map(len, split_sequences))


def shorten_trajectory_fully(trajectory):
    new_trajectory = []
    trajectory = [x for x in trajectory if x != 0]
    for i in range(len(trajectory)):
        if i == 0 or trajectory[i] != new_trajectory[-1]:
            new_trajectory.append(trajectory[i])
    return new_trajectory
        

# data1 = [1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 5, 5, 10]
# data = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
# data = [5, 10, 3, 6, 12, 17, 14, 4, 9, 13, 10, 8, 11, 2, 7, 10, 9, 13, 16, 10, 9, 11, 13, 3, 15, 5, 1, 16, 6, 10, 11, 8]
# print(data)
# data = [1, 2, 3, 4, 4, 4, 4, 4]
# print(resize(data))
# print(make_action_frequency_vector(data))