
def shorten_trajectory(trajectory):
    new_trajectory = []
    # remove 0's from the start of the trajectory
    while len(trajectory) > 1 and trajectory[0] == 0:
        trajectory.pop(0)
    # remove 0's from the end of the trajectory
    while len(trajectory) > 1 and trajectory[-1] == 0:
        trajectory.pop()
    for i in range(len(trajectory)):
        if i == 0 or trajectory[i] != new_trajectory[-1]:
            new_trajectory.append(trajectory[i])
    return new_trajectory