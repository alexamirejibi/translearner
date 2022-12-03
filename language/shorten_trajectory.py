
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
    # remove 0's from the start of the trajectory
    while len(trajectory) > 1 and trajectory[0] == 0:
        trajectory.pop(0)
    # remove 0's from the end of the trajectory
    while len(trajectory) > 1 and trajectory[-1] == 0:
        trajectory.pop(-1)

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
        

data = [1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 5, 5, 10]
print(shorten_trajectory(data))