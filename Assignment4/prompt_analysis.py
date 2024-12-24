import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pickle
import statistics
import numpy as np
import pandas as pd
import seaborn as sn


all_records_1_15 = pickle.load(
    open(f"./jina-embeddings-v2-base-en_llama3.2-1b_statistics_1-15.pickle", "rb")
)
all_records_16_17 = pickle.load(
    open(f"./jina-embeddings-v2-base-en_llama3.2-1b_statistics_16-17.pickle", "rb")
)
all_records_stella = pickle.load(
    open(f"./stella_en_1.5B_v5_llama3.2-1b_statistics.pickle", "rb")
)
all_records_diff = pickle.load(open(f"./diff_retriever_statistics.pickle", "rb"))
print(all_records_stella)
all_records = {
    k: v
    for record_pack in [all_records_1_15, all_records_16_17, all_records_diff]
    for k, v in record_pack.items()
}
print(all_records.keys())
print(all_records)


heatmap_raw = np.array([all_records[i] for i in range(1, 18)]).T
# shihtl> Heatmap of Prompts in Each Questions
# plt.figure(figsize=(13, 6.5))
# plt.title("Heatmap of Prompts in Each Questions", fontsize=14)
# sn.heatmap(
#     heatmap_raw,
#     annot=True,
#     vmin=0,
#     vmax=100,
#     fmt="3d",
#     cmap="crest",
#     xticklabels=np.array(range(1, 18)),
#     yticklabels=np.array(range(1, 11)),
# )
# plt.xlabel("Prompts", fontsize=12)
# plt.ylabel("Questions", fontsize=12)
# plt.savefig("./prompts_heatmap.png", bbox_inches="tight")

print(np.mean(heatmap_raw.T, axis=1))


# shihtl> avg
avg_records = {}
for k, v in all_records.items():
    if "accuracy" in str(k):
        print(f"{k.replace('_accuracy', '')}: {statistics.mean(v)}")
        avg_records[k.replace("_accuracy", "")] = statistics.mean(v)
print(avg_records)

# shihtl> avg graph
# title = "Compare Different Search Types and Retrieval Model Accuracy"
# plt.figure(figsize=(16 / 2.5, 9 / 2.5))

# x = [3, 4, 5]
# y1 = [avg_records[f"mmr_{i}_5"] for i in range(3, 6)]
# y2 = [avg_records[f"similarity_{i}"] for i in range(3, 6)]
# y3 = [avg_records[f"bm25_{i}"] for i in range(3, 6)]

# x_major_locator = MultipleLocator(1)

# ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)

# plt.plot(x, y1, label=f"Chroma mmr k=*", marker="o")
# plt.plot(x, y2, label=f"Chroma similarity k=*", marker="o")
# plt.plot(x, y3, label=f"BM25 k=*", marker="o")

# plt.legend()
# plt.suptitle(title, fontsize=12)
# plt.title("Chroma fetch_k=5, round=100", fontsize=8)
# plt.xlabel("k")
# plt.ylabel("Average Accuracy")
# plt.savefig(title + ".png", bbox_inches="tight")


# shihtl> mmr avg graph
# title = "Compare Different fetch_k Model Accuracy"
# plt.figure(figsize=(16 / 2.5, 9 / 2.5))

# x = [5, 7, 9]
# y1 = [avg_records[f"mmr_3_{i}"] for i in x]

# x_major_locator = MultipleLocator(2, 1)

# ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)

# plt.plot(x, y1, label=f"Chroma mmr fetch_k=*", marker="o")

# plt.legend()
# plt.suptitle(title, fontsize=12)
# plt.title("Chroma k=3, round=100", fontsize=8)
# plt.xlabel("fetch_k")
# plt.ylabel("Average Accuracy")
# plt.savefig(title + ".png", bbox_inches="tight")
