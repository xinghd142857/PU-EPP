# -*- coding: utf-8 -*-
# Author: xinghd


from gensim.models import Word2Vec
import pandas as pd
import numpy as np

def seq_to_kmers(seq, k=3):
    N = len(seq)
    return [seq[i:i+k] for i in range(N - k + 1)]


class Corpus(object):
    def __init__(self, proteins, ngram):
        self.v = proteins
        self.ngram = ngram

    def __iter__(self):
        for sentence in self.v :
            yield seq_to_kmers(sentence, self.ngram)


def get_protein_embedding(model, protein):
    vec = np.zeros((len(protein), 100))
    for i, word in enumerate(protein):
        vec[i, ] = model.wv[word]
    return vec

def train_word2vec(proteins, path2save):
    sent_corpus = Corpus(proteins, 3)
    model = Word2Vec(window=5, min_count=1, workers=10) 
    model.build_vocab(sent_corpus)
    model.train(sent_corpus, epochs=30, total_examples=model.corpus_count)
    model.save(path2save)
