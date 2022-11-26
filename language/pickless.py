import pickle


with open('data/train_lang_data.pkl', 'rb') as f:
    data = pickle.load(f)
    # for i in data[0]:
    #     print(i + '\n')
    print(data[0])
