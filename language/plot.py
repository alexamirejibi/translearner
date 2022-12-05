import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# load numpy array from npy file
data = np.load('data/train_log/task-4-lang-False.npy')
print(data)
# add data to plot

fig, ax = plt.subplots()  # Create a figure containing a single axes.
plt.plot(data.T[0], data.T[1], label='test')
plt.show()
#ax.plot(np.array([1, 2, 3, 4]), np.array([1, 4, 10, 50]));  # Plot some data on the axes.
#ax.plot([1, 2, 3, 4], [4, 3, 2, 1]);  # Plot some data on the axes.

# # display the plot
# plt.show()

# make empty 2d numpy array
# successes_array = np.empty((0,2), int)

# # add a row to the array
# a = np.array([[1, 2]])
# b = np.array([[5, 7]])
# successes_array = np.concatenate((successes_array, a), axis=0)

# print(successes_array)
# successes_array = np.concatenate((successes_array, b), axis=0)
# print(successes_array)

# print(successes_array[1][0])