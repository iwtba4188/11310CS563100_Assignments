import matplotlib.pyplot as plt
import pickle
import statistics
import numpy as np
import pandas as pd
import seaborn as sn


all_records_1_6 = pickle.load(open(f"./all_records_1-6.pkl", "rb"))
all_records_7 = pickle.load(open(f"./all_records_7.pkl", "rb"))
all_records = {
    k: v
    for record_pack in [all_records_1_6, all_records_7]
    for k, v in record_pack.items()
}
print(all_records.keys())
print(all_records["MultiLabelModel1_ep0_lr3e-05_bs8"])

spearman = []
accuracy = []
f1_score = []
for model_idx in range(1, 8):
    spearman.append(
        round(all_records[f"MultiLabelModel{model_idx}_lr3e-05_bs8"]["spearman"], 4)
    )
    accuracy.append(
        round(all_records[f"MultiLabelModel{model_idx}_lr3e-05_bs8"]["accuracy"], 4)
    )
    f1_score.append(
        round(all_records[f"MultiLabelModel{model_idx}_lr3e-05_bs8"]["f1_score"], 4)
    )

spearman = np.array(spearman)
accuracy = np.array(accuracy)
f1_score = np.array(f1_score)

matrices = np.array([spearman, accuracy, f1_score])

# print(spearman)
# print(accuracy)
# print(f1_score)
# print(matrices)
# print(matrices.T)

# shihtl> Confusion matrix of Model 3
test_confusion = all_records[f"MultiLabelModel3_lr3e-05_bs8"]["confusion"]
valid_confusion = all_records[f"MultiLabelModel3_ep9_lr3e-05_bs8"]["confusion"]

plt.figure(figsize=(6, 5))
title = "Confusion Matrix of Model 3 in Testing Dataset"
plt.title(title, fontsize=12)
df_test_confusion = pd.DataFrame(test_confusion, index=[1, 2, 3], columns=[1, 2, 3])
sn.heatmap(test_confusion, annot=True, cmap="crest", fmt="")
plt.savefig(title + ".png", bbox_inches="tight")

plt.figure(figsize=(6, 5))
title = "Confusion Matrix of 10th Epoch Model 3 in Validation Dataset"
plt.title(title, fontsize=10)
df_valid_confusion = pd.DataFrame(valid_confusion, index=[1, 2, 3], columns=[1, 2, 3])
sn.heatmap(valid_confusion, annot=True, cmap="crest", fmt="")
plt.savefig(title + ".png", bbox_inches="tight")
input()

# shihtl> Accuracy; model1: vanilla model, model6: two bert model
title = "Compare Multi-output and Separately Model Accuracy"
plt.figure(figsize=(16 / 2.5, 9 / 2.5))

x = range(1, 11)
y1 = [all_records[f"MultiLabelModel1_ep{ep-1}_lr3e-05_bs8"]["accuracy"] for ep in x]
y2 = [all_records[f"MultiLabelModel6_ep{ep-1}_lr3e-05_bs8"]["accuracy"] for ep in x]

plt.plot(x, y1, label="Multi-output model", marker="o")
plt.plot(x, y2, label="Separately model", marker="o")

plt.legend()
plt.suptitle(title, fontsize=12)
plt.title(
    "random_seed=814750857, learning_rate=3e-5, epoch=10, batch_size=8", fontsize=8
)
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.savefig(title + ".png", bbox_inches="tight")

# shihtl> Spearman; model1: vanilla model, model6: two bert model
title = "Compare Multi-output and Separately Model Spearman"
plt.figure(figsize=(16 / 2.5, 9 / 2.5))

x = range(1, 11)
y1 = [all_records[f"MultiLabelModel1_ep{ep-1}_lr3e-05_bs8"]["spearman"] for ep in x]
y2 = [all_records[f"MultiLabelModel6_ep{ep-1}_lr3e-05_bs8"]["spearman"] for ep in x]

plt.plot(x, y1, label="Multi-output model", marker="o")
plt.plot(x, y2, label="Separately model", marker="o")

plt.legend()
plt.suptitle(title, fontsize=12)
plt.title(
    "random_seed=814750857, learning_rate=3e-5, epoch=10, batch_size=8", fontsize=8
)
plt.xlabel("Epoch")
plt.ylabel("Validation Spearman Corr")
plt.savefig(title + ".png", bbox_inches="tight")

# shihtl> F1-score; model1: vanilla model, model6: two bert model
title = "Compare Multi-output and Separately Model F1-score"
plt.figure(figsize=(16 / 2.5, 9 / 2.5))

x = range(1, 11)
y1 = [all_records[f"MultiLabelModel1_ep{ep-1}_lr3e-05_bs8"]["f1_score"] for ep in x]
y2 = [all_records[f"MultiLabelModel6_ep{ep-1}_lr3e-05_bs8"]["f1_score"] for ep in x]

plt.plot(x, y1, label="Multi-output model", marker="o")
plt.plot(x, y2, label="Separately model", marker="o")

plt.legend()
plt.suptitle(title, fontsize=12)
plt.title(
    "random_seed=814750857, learning_rate=3e-5, epoch=10, batch_size=8", fontsize=8
)
plt.xlabel("Epoch")
plt.ylabel("Validation F1-score")
plt.savefig(title + ".png", bbox_inches="tight")

# shihtl> Avg total loss; model1: vanilla model, model6: two bert model
title = "Compare Average Training Total Loss Under Different Method"
plt.figure(figsize=(16 / 2.5, 9 / 2.5))

x = range(1, 11)
y1 = [
    statistics.fmean(
        all_records[f"MultiLabelModel1_ep{ep-1}_lr3e-05_bs8"]["loss_record_combined"]
    )
    for ep in x
]
y2 = [
    statistics.fmean(
        all_records[f"MultiLabelModel6_ep{ep-1}_lr3e-05_bs8"]["loss_record_combined"]
    )
    for ep in x
]

plt.plot(x, y1, label="Multi-output model", marker="o")
plt.plot(x, y2, label="Separately model", marker="o")

plt.legend()
plt.suptitle(title, fontsize=12)
plt.title(
    "random_seed=814750857, learning_rate=3e-5, epoch=10, batch_size=8", fontsize=8
)
plt.xlabel("Epoch")
plt.ylabel("Average Training Total Loss")
plt.savefig(title + ".png", bbox_inches="tight")

# shihtl> Avg reg loss; model1: vanilla model, model6: two bert model
title = "Compare Average Training Reg Loss Under Different Method"
plt.figure(figsize=(16 / 2.5, 9 / 2.5))

x = range(1, 11)
y1 = [
    statistics.fmean(
        all_records[f"MultiLabelModel1_ep{ep-1}_lr3e-05_bs8"]["loss_record_reg"]
    )
    for ep in x
]
y2 = [
    statistics.fmean(
        all_records[f"MultiLabelModel6_ep{ep-1}_lr3e-05_bs8"]["loss_record_reg"]
    )
    for ep in x
]

plt.plot(x, y1, label="Multi-output model", marker="o")
plt.plot(x, y2, label="Separately model", marker="o")

plt.legend()
plt.suptitle(title, fontsize=12)
plt.title(
    "random_seed=814750857, learning_rate=3e-5, epoch=10, batch_size=8", fontsize=8
)
plt.xlabel("Epoch")
plt.ylabel("Average Training Reg Loss")
plt.savefig(title + ".png", bbox_inches="tight")

# shihtl> Avg Class loss; model1: vanilla model, model6: two bert model
title = "Compare Average Training Class Loss Under Different Method"
plt.figure(figsize=(16 / 2.5, 9 / 2.5))

x = range(1, 11)
y1 = [
    statistics.fmean(
        all_records[f"MultiLabelModel1_ep{ep-1}_lr3e-05_bs8"]["loss_record_class"]
    )
    for ep in x
]
y2 = [
    statistics.fmean(
        all_records[f"MultiLabelModel6_ep{ep-1}_lr3e-05_bs8"]["loss_record_class"]
    )
    for ep in x
]

plt.plot(x, y1, label="Multi-output model", marker="o")
plt.plot(x, y2, label="Separately model", marker="o")

plt.legend()
plt.suptitle(title, fontsize=12)
plt.title(
    "random_seed=814750857, learning_rate=3e-5, epoch=10, batch_size=8", fontsize=8
)
plt.xlabel("Epoch")
plt.ylabel("Average Training Class Loss")
plt.savefig(title + ".png", bbox_inches="tight")

# shihtl> Performance in test dataset; model1: vanilla model, model6: two bert model
# title = "Compare Multi-output and Separately Model Performance"
# plt.figure(figsize=(16 / 2.5, 9 / 2.5))

# x_axis = range(len(x))
# width = 0.2  # the width of the bars

# plt.bar([p - width / 2 for p in x_axis], y1, width=width, label="Multi-output model")
# plt.bar([p + width / 2 for p in x_axis], y2, width=width, label="Separately model")

# plt.legend()
# plt.suptitle(title, fontsize=12)
# plt.title(
#     "random_seed=814750857, learning_rate=3e-5, epoch=10, batch_size=8", fontsize=8
# )

# plt.xticks(x_axis, x)
# plt.savefig(title + ".png", bbox_inches="tight")
###
# species = ["spearman", "accuracy", "f1_score"]
# penguin_means = {
#     f"Model {model_idx}": [
#         all_records[f"MultiLabelModel{model_idx}_lr3e-05_bs8"][slot] for slot in species
#     ]
#     for model_idx in range(1, 8)
# }

# x = np.arange(len(species))  # the label locations
# width = 0.1  # the width of the bars
# multiplier = 0

# fig, ax = plt.subplots(layout="constrained")

# for attribute, measurement in penguin_means.items():
#     offset = width * multiplier
#     rects = ax.bar(x + offset, measurement, width, label=attribute)
#     ax.bar_label(rects, padding=3)
#     multiplier += 1

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel("Length (mm)")
# ax.set_title("Penguin attributes by species")
# ax.set_xticks(x + width, species)
# ax.legend(loc="upper left", ncols=3)

# plt.show()

# shihtl> Accuracy; all models
title = "Compare Validation Accuracy of Different Models"
plt.figure(figsize=(16 / 2.5, 9 / 2.5))

x = range(1, 11)
for model_idx in range(1, 8):
    y = [
        all_records[f"MultiLabelModel{model_idx}_ep{ep-1}_lr3e-05_bs8"]["accuracy"]
        for ep in x
    ]

    plt.plot(x, y, label=f"Model {model_idx}", marker="o")

plt.legend(prop={"size": 9})
plt.suptitle(title, fontsize=12)
plt.title(
    "random_seed=814750857, learning_rate=3e-5, epoch=10, batch_size=8", fontsize=8
)
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.savefig(title + ".png", bbox_inches="tight")

# shihtl> Spearman; all models
title = "Compare Validation Spearman Corr of Different Models"
plt.figure(figsize=(16 / 2.5, 9 / 2.5))

x = range(1, 11)
for model_idx in range(1, 8):
    y = [
        all_records[f"MultiLabelModel{model_idx}_ep{ep-1}_lr3e-05_bs8"]["spearman"]
        for ep in x
    ]

    plt.plot(x, y, label=f"Model {model_idx}", marker="o")

plt.legend(prop={"size": 6})
plt.suptitle(title, fontsize=12)
plt.title(
    "random_seed=814750857, learning_rate=3e-5, epoch=10, batch_size=8", fontsize=8
)
plt.xlabel("Epoch")
plt.ylabel("Validation Spearman Corr")
plt.savefig(title + ".png", bbox_inches="tight")

# shihtl> F1-score; all models
title = "Compare Validation F1-score of Different Models"
plt.figure(figsize=(16 / 2.5, 9 / 2.5))

x = range(1, 11)
for model_idx in range(1, 8):
    y = [
        all_records[f"MultiLabelModel{model_idx}_ep{ep-1}_lr3e-05_bs8"]["f1_score"]
        for ep in x
    ]

    plt.plot(x, y, label=f"Model {model_idx}", marker="o")

plt.legend(prop={"size": 8})
plt.suptitle(title, fontsize=12)
plt.title(
    "random_seed=814750857, learning_rate=3e-5, epoch=10, batch_size=8", fontsize=8
)
plt.xlabel("Epoch")
plt.ylabel("Validation F1-score")
plt.savefig(title + ".png", bbox_inches="tight")
