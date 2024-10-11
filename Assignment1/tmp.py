import random
import math
import logging
from itertools import product

import pandas as pd
import numpy as np
import gensim.downloader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from nltk.tokenize.treebank import TreebankWordTokenizer
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")


class SentenceIter:
    def __init__(self, file_name):
        self.file_name = file_name

    def __next__(self):
        l = WordNetLemmatizer()

        with open(self.file_name, "r", encoding="utf-8") as file:
            for sentence in file.readline():
                # print(sentence)
                yield list(map(l.lemmatize, sentence.split()))


RANDOM_SEED = 38457190
RANDOM_PERCENT = 0.1

FILE_PATH = f"./wiki_texts_random_{RANDOM_SEED}_{RANDOM_PERCENT}_preprocessed.txt"

count = 0
for line in pre_next_line(FILE_PATH):

    print(count, end="\n\n\n")

    count += 1
    if count >= 10:
        break
