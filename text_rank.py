import numpy as np
import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
nltk.download('punkt')
model = api.load("glove-wiki-gigaword-100")


def sentence_vectors(sentences, model):
    vectors = []
    for sent in sentences:
        words = [w for w in word_tokenize(sent.lower()) if w.isalpha()]
        word_vectors = [model[word] for word in words if word in model]
        if len(word_vectors) > 0:
            v = sum(word_vectors) / len(word_vectors)
        else:
            v = np.zeros((model.vector_size,))
        vectors.append(v)
    return np.array(vectors)


def rank(text):
    sentences = sent_tokenize(text)
    sentence_embeddings = sentence_vectors(sentences, model)
    similarity_matrix = cosine_similarity(sentence_embeddings)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    for i, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        print(f"Sentença {i}: {sentences[i]} \nPontuação: {score}\n")



