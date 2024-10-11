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
from nltk.stem import WordNetLemmatizer


class SentenceIter:
    def __init__(self, file_name):
        self.file_name = file_name

    def gen(self):
        l = WordNetLemmatizer()

        with open(self.file_name, "r", encoding="utf-8") as file:
            for sentence in file.readline():
                # print(sentence)
                yield list(map(l.lemmatize, sentence.split()))

    def __iter__(self):
        return self.gen()

    def __next__(self):
        result = next(self.gen)

        logging.info(result)

        if result is None:
            raise StopIteration
        else:
            return result


def process(epoch, vector_size):
    # shihtl> Settings for model
    RANDOM_SEED = 38457190
    RANDOM_PERCENT = 0.1
    EPOCH = epoch
    VECTOR_SIZE = vector_size
    PARAMS_INFO = f"{RANDOM_SEED}_{RANDOM_PERCENT}_{EPOCH}_{VECTOR_SIZE}_LEMMA"

    USE_PRETRAIN_MODEL = False
    SAVE_MODEL_PATH = f"./models/wiki_texts_random_{PARAMS_INFO}.model"
    LOAD_MODEL_PATH = f"./models/wiki_texts_random_{PARAMS_INFO}.model"
    FILE_PATH = f"./wiki_texts_random_{RANDOM_SEED}_{RANDOM_PERCENT}_preprocessed.txt"

    if USE_PRETRAIN_MODEL:
        my_model = Word2Vec.load(LOAD_MODEL_PATH)
    else:
        logging.basicConfig(level=logging.INFO)
        my_model = Word2Vec(
            SentenceIter(FILE_PATH), vector_size=VECTOR_SIZE, epochs=EPOCH
        )
        my_model.save(SAVE_MODEL_PATH)

    data = pd.read_csv("questions-words.csv")

    # Do predictions and preserve the gold answers (word_D)
    preds = []
    golds = []

    # shihtl> preds_cache
    try:
        cache = pd.read_csv(f"preds_cache_{PARAMS_INFO}_my_model.csv")
        preds = list(cache.preds)
        golds = list(cache.golds)
    except:
        for analogy in tqdm(data["Question"]):
            # TODO6: Write your code here to use your trained word embeddings for getting predictions of the analogy task.
            # You should also preserve the gold answers during iterations for evaluations later.
            """Hints
            # Unpack the analogy (e.g., "man", "woman", "king", "queen")
            # Perform vector arithmetic: word_b + word_c - word_a should be close to word_d
            # Source: https://github.com/piskvorky/gensim/blob/develop/gensim/models/keyedvectors.py#L776
            # Mikolov et al., 2013: big - biggest and small - smallest
            # Mikolov et al., 2013: X = vector(”biggest”) − vector(”big”) + vector(”small”).
            """
            word_a, word_b, word_c, word_d = map(str.lower, analogy.split())
            this_pred = my_model.wv.most_similar([word_b, word_c], word_a, topn=1)[0][0]

            preds.append(this_pred)
            golds.append(word_d)

        # shihtl> preds_cache
        preds_cache = pd.DataFrame({"preds": preds, "golds": golds})
        preds_cache.to_csv(f"preds_cache_{PARAMS_INFO}_my_model.csv", index=False)

    # Perform evaluations. You do not need to modify this block!!

    def calculate_accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
        return np.mean(gold == pred)

    golds_np, preds_np = np.array(golds), np.array(preds)
    data = pd.read_csv("questions-words.csv")

    # Evaluation: categories
    for category in data["Category"].unique():
        mask = data["Category"] == category
        golds_cat, preds_cat = golds_np[mask], preds_np[mask]
        acc_cat = calculate_accuracy(golds_cat, preds_cat)
        logging.info(f"Category: {category}, Accuracy: {acc_cat * 100}%")

    # Evaluation: sub-categories
    for sub_category in data["SubCategory"].unique():
        mask = data["SubCategory"] == sub_category
        golds_subcat, preds_subcat = golds_np[mask], preds_np[mask]
        acc_subcat = calculate_accuracy(golds_subcat, preds_subcat)
        logging.info(f"Sub-Category{sub_category}, Accuracy: {acc_subcat * 100}%")

    # Collect words from Google Analogy dataset
    SUB_CATEGORY = ": family"

    # TODO7: Plot t-SNE for the words in the SUB_CATEGORY `: family`
    sub_category_data = data[data.SubCategory == SUB_CATEGORY]
    sub_category_data_str = " ".join(sub_category_data.Question)
    words = np.unique(np.array(sub_category_data_str.split()))
    # print(words[:10])

    X = np.array([my_model.wv[word] for word in words])
    # print(X[0])

    embedded = TSNE().fit_transform(X)
    # print(embedded[:5, :])

    plt.figure(figsize=(16, 12))
    plt.scatter(embedded[:, 0], embedded[:, 1])

    for idx, dots in enumerate(embedded):
        plt.annotate(words[idx], (dots[0] + 0.04, dots[1]))

    plt.title(f"Word Relationships from Google Analogy Task (My Model {PARAMS_INFO})")
    plt.savefig(f"word_relationships_{PARAMS_INFO}_my_model.png", bbox_inches="tight")
    # plt.show()


p_epoch = [5]
p_vector_size = [50]
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="training.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S",
)
for epoch, vector_size in list(product(p_epoch, p_vector_size)):
    try:
        logging.info(f"Starting process with hyperparams: {epoch=}, {vector_size=}")
        process(epoch=epoch, vector_size=vector_size)
    except Exception as e:
        logging.warning(f"Something wrong: {e}")
