
def shorten_trajectory2(trajectory):
    new_trajectory = []
    # remove 0's from the start of the trajectory
    # while len(trajectory) > 1 and trajectory[0] == 0:
    #     trajectory.pop(0)
    # # remove 0's from the end of the trajectory
    # while len(trajectory) > 1 and trajectory[-1] == 0:
    #     trajectory.pop()
    for i in range(len(trajectory)):
        if i == 0 or trajectory[i] != new_trajectory[-1]:
            new_trajectory.append(trajectory[i])
    return new_trajectory


def shorten_trajectory(trajectory):
    max_traj_length = 8
    # # remove 0's from the start of the trajectory
    # while len(trajectory) > 1 and trajectory[0] == 0:
    #     trajectory.pop(0)
    # # remove 0's from the end of the trajectory
    # while len(trajectory) > 1 and trajectory[-1] == 0:
    #     trajectory.pop(-1)
    
    # remove all 0s
    trajectory = [i for i in trajectory if i != 0]
    if len(trajectory) <= max_traj_length:
        return trajectory
    
    traj_len = len(trajectory)
    shrink_factor = max_traj_length / traj_len

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
        new_count = round(len(l) * shrink_factor)
        if new_count == 0:
            new_count = 1
        new_traj.extend(l[:new_count])

    return new_traj


def shorten_traj_recency(trajectory):
    max_traj_length = 20
    length = len(trajectory)

    # # remove 0's from the start of the trajectory
    # while len(trajectory) > 1 and trajectory[0] == 0:
    #     trajectory.pop(0)
    # # remove 0's from the end of the trajectory
    # while len(trajectory) > 1 and trajectory[-1] == 0:
    #     trajectory.pop(-1)

    # remove all 0s
    trajectory = [i for i in trajectory if i != 0]
    if len(trajectory) <= max_traj_length:
        return trajectory

    traj_len = len(trajectory)

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
        significance_coefficients.append(max(0.3, 1 - (1 / (i+1))))

    new_split_sequences = split_sequences.copy()
    #print(len(new_split_sequences), len(significance_coefficients))
    while find_total_length(new_split_sequences) > max_traj_length:
        length = find_total_length(new_split_sequences)
        #print('yes')
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
            break

    for l in new_split_sequences:
        new_traj.extend(l)

    return new_traj

def expand_trajectory(trajectory):
    max_traj_length = 20
    length = len(trajectory)

    # # remove 0's from the start of the trajectory
    # while len(trajectory) > 1 and trajectory[0] == 0:
    #     trajectory.pop(0)
    # # remove 0's from the end of the trajectory
    # while len(trajectory) > 1 and trajectory[-1] == 0:
    #     trajectory.pop(-1)

    # remove all 0s
    # trajectory = [i for i in trajectory if i != 0]
    # if len(trajectory) <= max_traj_length:
    #     return trajectory

    traj_len = len(trajectory)

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
    #print(len(new_split_sequences), len(significance_coefficients))
    while find_total_length(new_split_sequences) < max_traj_length:
        #length = find_total_length(new_split_sequences)
        #print('yes')
        for i in range(len(new_split_sequences)):
            new_count = round(len(new_split_sequences[i]) * significance_coefficients[i])
            # if new_count == 0:
            #     new_count = 1
            # extend the sequence by repeating the last element
            new_split_sequences[i] = [] + [new_split_sequences[i][0]] * new_count
        # if find_total_length(new_split_sequences) == length and length > max_traj_length:
        #     shrink_factor = max_traj_length / find_total_length(new_split_sequences)
        #     for i in range(len(new_split_sequences)):
        #         new_count = round(len(new_split_sequences[i]) * shrink_factor)
        #         # if new_count == 0:
        #         #     new_count = 1
        #         new_split_sequences[i] = new_split_sequences[i][:new_count]
        #     break

    for l in new_split_sequences:
        new_traj.extend(l)

    return new_traj


def find_total_length(split_sequences):
    return sum(map(len, split_sequences))

    # # replace repeating sequences of actions with a single action
    # no_repeat = []
    # for i in range(len(trajectory)):
    #     if i == 0 or trajectory[i] != no_repeat[-1]:
    #         no_repeat.append(trajectory[i])

    # # Calculate the relative frequencies of each item in the list
    # counts = {}
    # for i in trajectory:
    #     counts[i] = counts.get(i, 0) + 1
    # total = len(trajectory)
    # proportions = {i: counts[i]/total for i in counts}
        

# data1 = [1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 5, 5, 10]
#data = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]

# print(shorten_traj_recency(data))

data = [1, 2, 3, 4, 5]
print(expand_trajectory(data))