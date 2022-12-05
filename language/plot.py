import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np



# load numpy array from npy file
task = 4
data_lang = np.load('data/final_results/task-{}-lang-true.npy'.format(task))
data_ext = np.load('data/final_results/task-{}-lang-false.npy'.format(task))

# while the 0th dimension of data_ext doesn't match the 0th dimension of data_lang, remove the last element of data_ext
# make the two arrays end at the same 0th index
#data_ext = data_ext[0:len(data_lang)]
print(data_lang[-1][0], data_ext[-1][0])

while data_lang[-1][0] != data_ext[-1][0]:
    while data_lang[-1][0] < data_ext[-1][0]:
        data_ext = data_ext[:-1]

    while data_lang[-1][0] > data_ext[-1][0]:
        data_lang = data_lang[:-1]

print(data_lang[-1][0], data_ext[-1][0])

print('success rate lang: ', (data_lang[-1][1]/data_lang[-1][0]) * 100)
print('success rate ext: ', (data_ext[-1][1]/data_ext[-1][0]) * 100)


# def make_same_size(data1, data2):
#     if len(data1) > len(data2):
#         data1 = data1[:len(data2)]
#     else:
#         data2 = data2[:len(data1)]
#     return data1, data2


fig, ax = plt.subplots()  # Create a figure containing a single axes.
plt.plot(data_lang.T[0], data_lang.T[1], label='Language + external', color='red')
plt.plot(data_ext.T[0], data_ext.T[1], label='Only external', color='blue')
#plt.annotate('lang', (data_lang.T[0][-1], data_lang.T[1][-1]), color='red')
# draw labels on axis and set title
ax.set_xlabel('Timesteps')
ax.set_ylabel('Reward')
ax.set_title('Task {}: Reward vs Timesteps'.format(task))
# add subtitle
ax.legend()
# color areas under curve
ax.fill_between(data_lang.T[0], data_lang.T[1], color='red', alpha=0.2)
ax.fill_between(data_ext.T[0], data_ext.T[1], color='blue', alpha=0.2)

# annotate the max values of each curve
ax.annotate(data_lang[-1][-1], (data_lang.T[0][-1], data_lang.T[1][-1]), color='red')
ax.annotate(data_ext[-1][-1], (data_ext.T[0][-1], data_ext.T[1][-1]), color='blue')

#plt.show()

# save plot info 
#fig.savefig('data/figures/task-{}.png'.format(task))
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