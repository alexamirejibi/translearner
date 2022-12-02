import torch
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# make 15 random sentences
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.',
    'I am a sentence for which I would like to get its embedding',
    'This is very similar to other sentence embeddings frameworks',
    'However, BERT was pretrained using a masked language modeling (MLM) objective',
    'MLM is a fill-in-the-blank task where a model uses the context words surrounding a [MASK] token to try to predict what word should be there',
    'This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.',
    'I am a sentence for which I would like to get its embedding',
    'This is very similar to other sentence embeddings frameworks',
    'However, BERT was pretrained using a masked language modeling (MLM) objective',]
sentence_embeddings = model.encode(sentences)

for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")

# print the dimensionality of the embeddings
print("Dimensionality of the embeddings:", len(sentence_embeddings[0])) # 384


# # visualize the embeddings
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # reduce the dimensionality of the embeddings to 2D
# tsne = TSNE(n_components=2, perplexity=5)
# embeddings_2d = tsne.fit_transform(sentence_embeddings)

# # plot the embeddings
# plt.figure(figsize=(16, 9))
# sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=sentences, legend='full', alpha=0.3)
# plt.show()

    