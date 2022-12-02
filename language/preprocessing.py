import numpy as np
import string
import os

def load_actions(data_dir):
    clip_to_actions = {}
    with open(os.path.join(data_dir, 'action_labels.txt')) as f:
        for line in f.readlines():
            line = line.strip()
            parts = line.split()
            clip_id = parts[0]
            actions = map(eval, parts[1:])
            clip_to_actions[clip_id] = list(actions)
    return clip_to_actions


def load_annotations_file(filename):
    clip_ids = []
    sentences = []
    translator = str.maketrans('', '', string.punctuation)
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip()
            (clip_id, sentence) = line.split('\t')
            sentence = sentence.lower()
            sentence = sentence.translate(translator)
            clip_ids.append(clip_id.strip())
            sentences.append(sentence)
    return clip_ids, sentences




# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# labels = {'unrelated':0,
#           'related':1,
#           }

# tokenizer = BertTokenizer.from_pretrained(
#     'bert-base-uncased',
#     do_lower_case = True
#     )

    # for i in data[0]:
    #     print(i + '\n')
    # print values coresponding to the key "sentence
    # for i in data:
    #     print(i['sentence'] + '\t' + i['clip_id'])
    # print(data[0]['sentence'] + '\t' + data[0]['clip_id'])
    # print(data[0])
    #ids, sentences = load_annotations_file('data/annotations.txt')
    # for i, s in zip(ids, sentences):
    #     print(i, s)
    # id = data[0]['clip_id']
    # zipped = zip(data[0]['sentence'], clip_to_actions[id])
    # print(zipped)
    #sentence_action_pairs = list(zip(data['sentence']))
    # for i in data:
    #     sentence_action_pairs += list(zip(i['sentence'], clip_to_actions[i['clip_id']]))
    # print(sentence_action_pairs[1])
    #print(data[])
    #print(data[0]['sentence'], clip_to_actions[id]) # 0,0,0,0,1...





# def main(data_dir, lang_enc_pretrained, output_dir):
#     train_clip_ids, train_sentences = load_annotations_file(os.path.join(
#         data_dir, 'train_annotations.txt'))
#     test_clip_ids, test_sentences = load_annotations_file(os.path.join(
#         data_dir, 'test_annotations.txt'))

#     infersent = InferSentFeatures(lang_enc_pretrained, train_sentences)
#     glove = GloveFeatures(lang_enc_pretrained)
#     onehot = OnehotFeatures(train_sentences)

#     infersent_embeddings_train = infersent.generate_embeddings(train_sentences)
#     glove_embeddings_train = glove.generate_embeddings(train_sentences)
#     onehot_embeddings_train = onehot.generate_embeddings(train_sentences)

#     train_data = []
#     for idx in range(len(train_clip_ids)):
#         clip_id = train_clip_ids[idx]
#         data_pt = {}
#         data_pt['clip_id'] = clip_id
#         data_pt['sentence'] = train_sentences[idx]
#         data_pt['glove'] = glove_embeddings_train[idx]
#         data_pt['infersent'] = infersent_embeddings_train[idx]
#         data_pt['onehot'] = onehot_embeddings_train[idx]
#         train_data.append(data_pt)
#     pickle.dump(train_data, open(os.path.join(output_dir, 'train_lang_data.pkl'), 'wb'))

#     infersent_embeddings_test = infersent.generate_embeddings(test_sentences)
#     glove_embeddings_test = glove.generate_embeddings(test_sentences)
#     onehot_embeddings_test = onehot.generate_embeddings(test_sentences)

#     test_data = []
#     for idx in range(len(test_clip_ids)):
#         clip_id = test_clip_ids[idx]
#         data_pt = {}
#         data_pt['clip_id'] = clip_id
#         data_pt['sentence'] = test_sentences[idx]
#         data_pt['glove'] = glove_embeddings_test[idx]
#         data_pt['infersent'] = infersent_embeddings_test[idx]
#         data_pt['onehot'] = onehot_embeddings_test[idx]
#         test_data.append(data_pt)
#     pickle.dump(test_data, open(os.path.join(output_dir, 'test_lang_data.pkl'), 'wb'))


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_dir', type=str, default='./data',
#         help='path to data directory')
#     parser.add_argument('--output_dir', type=str, default='./data',
#         help='directory where processed files will be saved')
#     parser.add_argument('--lang_enc_dir', type=str, default='./lang_enc_pretrained',
#         help='directory for pretrained language encoder models')
#     args = parser.parse_args()
#     main(args.data_dir, args.lang_enc_dir, args.output_dir)

