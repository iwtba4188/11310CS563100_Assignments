import pandas as pd
import pandas as pd
import numpy as np
import gensim.downloader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from gensim.models import Word2Vec
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="training.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

data = pd.read_csv("questions-words.csv")


def eval_pred(random_seed, random_percent, epoch, vector_size):
    global data

    params_info = f"{random_seed}_{random_percent}_{epoch}_{vector_size}"

    # Do predictions and preserve the gold answers (word_D)
    preds = []
    golds = []

    # shihtl> preds_cache
    cache = pd.read_csv(f"preds_cache_{params_info}_my_model.csv")
    preds = list(cache.preds)
    golds = list(cache.golds)

    # Perform evaluations. You do not need to modify this block!!
    def calculate_accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
        return np.mean(gold == pred)

    golds_np, preds_np = np.array(golds), np.array(preds)
    data = pd.read_csv("questions-words.csv")

    res_accu = {}

    # Evaluation: categories
    for category in data["Category"].unique():
        mask = data["Category"] == category
        golds_cat, preds_cat = golds_np[mask], preds_np[mask]
        acc_cat = calculate_accuracy(golds_cat, preds_cat)
        # print(f"Category: {category}, Accuracy: {acc_cat * 100}%")
        res_accu[category] = round(acc_cat * 100, 3)

    # Evaluation: sub-categories
    for sub_category in data["SubCategory"].unique():
        mask = data["SubCategory"] == sub_category
        golds_subcat, preds_subcat = golds_np[mask], preds_np[mask]
        acc_subcat = calculate_accuracy(golds_subcat, preds_subcat)
        # print(f"Sub-Category{sub_category}, Accuracy: {acc_subcat * 100}%")
        res_accu[sub_category] = round(acc_subcat * 100, 3)

    return res_accu


res_accu = {}
for random_seed, random_percent, epoch, vector_size in [
    [38457190, 0.1, 1, 50],
    [38457190, 0.1, 5, 50],
    [38457190, 0.1, 10, 10],
    [38457190, 0.1, 10, 30],
    [38457190, 0.1, 10, 50],
    [38457190, 0.2, 5, 50],
    [38457190, 0.35, 5, 50],
    [38457190, 0.5, 5, 50],
]:
    res_accu[(random_seed, random_percent, epoch, vector_size)] = eval_pred(
        random_seed, random_percent, epoch, vector_size
    )

# shihtl> print all accuracy
for key, val in res_accu.items():
    print(key)
    for kkey, vval in val.items():
        print(f"{kkey} {vval}")


# shihtl> 比較不同 percent
plt.figure(figsize=(16 / 2.5, 9 / 2.5))
x = [0.1, 0.2, 0.35, 0.5]
for feature in [
    "Semantic",
    "Syntatic",
    # ": capital-common-countries",
    # ": capital-world",
    # ": currency",
    # ": city-in-state",
    ": family",
    # ": gram1-adjective-to-adverb",
    # ": gram2-opposite",
    # ": gram4-superlative",
    # ": gram5-present-participle",
    # ": gram6-nationality-adjective",
    # ": gram7-past-tense",
    # ": gram8-plural",
    # ": gram9-plural-verbs",
]:
    y = [res_accu[(38457190, xx, 5, 50)][feature] for xx in x]
    print(feature)
    print(y)
    plt.plot(x, y, label=feature, marker="o")

plt.legend()
plt.suptitle("Compare Different Sample Percentage", fontsize=12)
plt.title("random_seed=38457190, epoch=5, vector_size=50", fontsize=8)
plt.xlabel("Sample Percentage")
plt.ylabel("Accuracy (%)")
plt.savefig("Compare Different Sample Percentage.png", bbox_inches="tight")
# plt.show()


# shihtl> 比較不同 vector_size
plt.figure(figsize=(16 / 2.5, 9 / 2.5))
x = [10, 30, 50]
for feature in [
    "Semantic",
    "Syntatic",
    # ": capital-common-countries",
    # ": capital-world",
    # ": currency",
    # ": city-in-state",
    ": family",
    # ": gram1-adjective-to-adverb",
    # ": gram2-opposite",
    # ": gram4-superlative",
    # ": gram5-present-participle",
    # ": gram6-nationality-adjective",
    # ": gram7-past-tense",
    # ": gram8-plural",
    # ": gram9-plural-verbs",
]:
    y = [res_accu[(38457190, 0.1, 10, xx)][feature] for xx in x]
    print(feature)
    print(y)
    plt.plot(x, y, label=feature, marker="o")

plt.legend()
plt.suptitle("Compare Different Vector Size", fontsize=12)
plt.title("random_seed=38457190, percentage=0.1, epoch=10", fontsize=8)
plt.xlabel("Vector Size")
plt.ylabel("Accuracy (%)")
plt.savefig("Compare Different Vector Size.png", bbox_inches="tight")
# plt.show()


# shihtl> 比較不同 epoch
plt.figure(figsize=(16 / 2.5, 9 / 2.5))
x = [1, 5, 10]
for feature in [
    "Semantic",
    "Syntatic",
    # ": capital-common-countries",
    # ": capital-world",
    # ": currency",
    # ": city-in-state",
    ": family",
    # ": gram1-adjective-to-adverb",
    # ": gram2-opposite",
    # ": gram4-superlative",
    # ": gram5-present-participle",
    # ": gram6-nationality-adjective",
    # ": gram7-past-tense",
    # ": gram8-plural",
    # ": gram9-plural-verbs",
]:
    y = [res_accu[(38457190, 0.1, xx, 50)][feature] for xx in x]
    print(feature)
    print(y)
    plt.plot(x, y, label=feature, marker="o")

plt.legend()
plt.suptitle("Compare Different Epoch", fontsize=12)
plt.title("random_seed=38457190, percentage=0.1, vector_size=50", fontsize=8)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.savefig("Compare Different Epoch", bbox_inches="tight")
plt.show()


# for idx in range(len(y_semantic)):
#     plt.annotate((x[idx], y_semantic[idx]), (x[idx] - 0.02, y_semantic[idx] + 0.04))
# for idx in range(len(y_syntatic)):
#     plt.annotate((x[idx], y_syntatic[idx]), (x[idx] - 0.02, y_syntatic[idx] + 0.04))
