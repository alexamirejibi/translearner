import pickle


with open('data/sentence_action_pairs.pkl', 'rb') as f:
    data = pickle.load(f)
    # for i in data[0]:
    #     print(i + '\n')
    # print values coresponding to the key "sentence
    for i in data:
        print(i['sentence'] + '\t' + i['trajectory'])
    # print(data[0]['sentence'] + '\t' + data[0]['clip_id'])
    # print(data[0])

# go past the lines to the left then jump when you reach the edge                 1340/clip_104.mp4
# jump to the rope climb up then jump to the middle platform and get the item     1289/clip_124.mp4
