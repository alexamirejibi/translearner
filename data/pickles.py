# open pickle
import pickle

with open('data/full_sent_traj_pairs.pkl', 'rb') as f:
    data = pickle.load(f)
    
print(data[0])