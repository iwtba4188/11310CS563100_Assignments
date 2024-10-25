import matplotlib.pyplot as plt
import pickle
import statistics


def template():
    """用來複製的模板，直接執行不會有結果"""
    plt.figure(figsize=(16 / 2.5, 9 / 2.5))

    plt.legend()
    plt.suptitle("Compare Different Sample Percentage", fontsize=12)
    plt.title("random_seed=38457190, epoch=5, vector_size=50", fontsize=8)
    plt.xlabel("Sample Percentage")
    plt.ylabel("Accuracy (%)")
    plt.savefig("Compare Different Sample Percentage.png", bbox_inches="tight")


accuracy_data = {
    "LSTM_2_1e-2": [0.08760493827160494, 0.09066666666666667, 0.09664197530864198],
    "LSTM_2_1e-3": [0.5942526115859449, 0.6952516619183285, 0.7290332383665717],
    "LSTM_2_1e-4": [0.4701766381766382, 0.5673428300094967, 0.6039316239316239],
    "LSTM_3_1e-3": [0.6131889838556506, 0.7272364672364673, 0.8123798670465338],
    "LSTM_4_1e-3": [0.6480987654320988, 0.7724786324786325, 0.8418727445394112],
    "RNN_3_1e-3": [0.11670655270655271, 0.12018993352326686, 0.10615384615384615],
    "RNN_3_1e-2": [0.004262108262108262, 0.003418803418803419, 0.004246913580246914],
    "GRU_3_1e-3": [0.50067616334283, 0.5658879392212726, 0.5705299145299145],
    "LSTM_3_1e-3_no_id_16": [
        0.6178997465850794,
        0.7013906916373076,
        0.7482539093887137,
    ],
    "LSTM_3_1e-3_vocab_by_freq_increasing": [
        0.609994301994302,
        0.7083722697056031,
        0.778412155745489,
    ],
    "LSTM_3_1e-3_vocab_by_freq_descending": [
        0.6245052231718898,
        0.7012117758784425,
        0.7469059829059829,
    ],
}

labels = [
    f"LSTM_2_{1e-2}",
    f"LSTM_2_{1e-3}",
    f"LSTM_2_{1e-4}",
    f"LSTM_3_{1e-3}",
    f"LSTM_4_{1e-3}",
    f"RNN_3_{1e-3}",
    f"RNN_3_{1e-2}",
    f"GRU_3_{1e-3}",
    f"LSTM_3_{1e-3}_no_id_16",
    f"LSTM_3_{1e-3}_vocab_by_freq_increasing",
    f"LSTM_3_{1e-3}_vocab_by_freq_descending",
]

avg_loss_data = {}
for label in labels:
    this_label_avg_loss = [
        statistics.fmean(
            pickle.load(open(f"./loss_data/{epoch}_{label}_loss_record.pkl", "rb"))
        )
        for epoch in ["1", "2", "3"]
    ]

    avg_loss_data[label] = this_label_avg_loss

# shihtl> 比較不同 learning rate 的 accuracy
plt.figure(figsize=(16 / 2.5, 9 / 2.5))

x = [1, 2, 3]  # shihtl> epoch

for learning_rate in ["1e-2", "1e-3", "1e-4"]:
    y = accuracy_data[f"LSTM_2_{learning_rate}"]
    plt.plot(x, y, label=learning_rate, marker="o")

plt.legend(loc="upper left")
plt.suptitle("Compare Validation Accuracy Under Different Learning Rate", fontsize=12)
plt.title("layer_type=LSTM, num_of_layers=2, vocab_dict=ASCII order", fontsize=8)
plt.xlabel("Epoch")
plt.xticks(x)
plt.ylabel("Validation Accuracy")
plt.ylim(0, 1)
plt.savefig(
    "Compare Validation Accuracy Under Different Learning Rate.png".replace(" ", "_"),
    bbox_inches="tight",
)

# shihtl> 比較不同 learning rate 的 loss
plt.figure(figsize=(16 / 2.5, 9 / 2.5))

x = [1, 2, 3]  # shihtl> epoch

for learning_rate_label, learning_rate in zip(
    ["1e-2", "1e-3", "1e-4"], [f"{1e-2}", f"{1e-3}", f"{1e-4}"]
):
    y = avg_loss_data[f"LSTM_2_{learning_rate}"]
    plt.plot(x, y, label=learning_rate_label, marker="o")

plt.legend(loc="center right")
plt.suptitle("Compare Average Training Loss Under Different Learning Rate", fontsize=12)
plt.title("layer_type=LSTM, num_of_layers=2, vocab_dict=ASCII order", fontsize=8)
plt.xlabel("Epoch")
plt.xticks(x)
plt.ylabel("Avg. Training Loss")
plt.savefig(
    "Compare Average Training Loss Under Different Learning Rate.png".replace(" ", "_"),
    bbox_inches="tight",
)

# shihtl> 比較不同結構 (RNN, GRU, LSTM) 的 accuracy
plt.figure(figsize=(16 / 2.5, 9 / 2.5))

x = [1, 2, 3]  # shihtl> epoch

for structure in ["RNN", "GRU", "LSTM"]:
    y = accuracy_data[f"{structure}_3_1e-3"]
    plt.plot(x, y, label=structure, marker="o")

plt.legend()
plt.suptitle("Compare Validation Accuracy Under Different RNN Structure", fontsize=12)
plt.title("learning_rate=1e-3, num_of_layers=3, vocab_dict=ASCII order", fontsize=8)
plt.xlabel("Epoch")
plt.xticks(x)
plt.ylabel("Validation Accuracy")
plt.ylim(0, 1)
plt.savefig(
    "Compare Validation Accuracy Under Different RNN Structure.png".replace(" ", "_"),
    bbox_inches="tight",
)

# shihtl> 比較不同結構 (RNN, GRU, LSTM) 的 loss
plt.figure(figsize=(16 / 2.5, 9 / 2.5))

x = [1, 2, 3]  # shihtl> epoch

for structure in ["RNN", "GRU", "LSTM"]:
    y = avg_loss_data[f"{structure}_3_{1e-3}"]
    plt.plot(x, y, label=structure, marker="o")

plt.legend(loc="center right")
plt.suptitle("Compare Average Training Loss Under Different RNN Structure", fontsize=12)
plt.title("learning_rate=1e-3, num_of_layers=3, vocab_dict=ASCII order", fontsize=8)
plt.xlabel("Epoch")
plt.xticks(x)
plt.ylabel("Avg. Training Loss")
plt.savefig(
    "Compare Average Training Loss Under Different RNN Structure.png".replace(" ", "_"),
    bbox_inches="tight",
)

# shihtl> 比較是否有看過 "9" (編號 16) 的 accuracy
plt.figure(figsize=(16 / 2.5, 9 / 2.5))

x = [1, 2, 3]  # shihtl> epoch

for label, id in zip(
    ['Contains "9"', 'NOT Contains "9"'],
    ["LSTM_3_1e-3", "LSTM_3_1e-3_no_id_16"],
):
    y = accuracy_data[id]
    plt.plot(x, y, label=label, marker="o")

plt.legend()
plt.suptitle('Compare Validation Accuracy Whether Training Has "9"', fontsize=12)
plt.title(
    "layer_type=LSTM, learning_rate=1e-3, num_of_layers=3, vocab_dict=ASCII order",
    fontsize=8,
)
plt.xlabel("Epoch")
plt.xticks(x)
plt.ylabel("Validation Accuracy")
plt.ylim(0, 1)
plt.savefig(
    "Compare Validation Accuracy Whether Training Has 9.png".replace(" ", "_"),
    bbox_inches="tight",
)

# shihtl> 比較是否有看過 "9" (編號 16) 的 loss
plt.figure(figsize=(16 / 2.5, 9 / 2.5))

x = [1, 2, 3]  # shihtl> epoch

for label, id in zip(
    ['Contains "9"', 'NOT Contains "9"'],
    [f"LSTM_3_{1e-3}", f"LSTM_3_{1e-3}_no_id_16"],
):
    y = avg_loss_data[id]
    plt.plot(x, y, label=label, marker="o")

plt.legend(loc="center right")
plt.suptitle('Compare Average Training Loss Whether Training Has "9"', fontsize=12)
plt.title("learning_rate=1e-3, num_of_layers=3, vocab_dict=ASCII order", fontsize=8)
plt.xlabel("Epoch")
plt.xticks(x)
plt.ylabel("Avg. Training Loss")
plt.savefig(
    "Compare Average Training Loss Whether Training Has 9.png".replace(" ", "_"),
    bbox_inches="tight",
)

# shihtl> 比較 LSTM 層數的 accuracy
plt.figure(figsize=(16 / 2.5, 9 / 2.5))

x = [1, 2, 3]  # shihtl> epoch

for num_of_layers in [2, 3, 4]:
    y = accuracy_data[f"LSTM_{num_of_layers}_1e-3"]
    plt.plot(x, y, label=num_of_layers, marker="o")

plt.legend()
plt.suptitle("Compare Validation Accuracy Under Different Layer Numbers", fontsize=12)
plt.title("layer_type=LSTM, learning_rate=1e-3, vocab_dict=ASCII order", fontsize=8)
plt.xlabel("Epoch")
plt.xticks(x)
plt.ylabel("Validation Accuracy")
plt.ylim(0, 1)
plt.savefig(
    "Compare Validation Accuracy Under Different Layer Numbers.png".replace(" ", "_"),
    bbox_inches="tight",
)

# shihtl> 比較 LSTM 層數的 loss
plt.figure(figsize=(16 / 2.5, 9 / 2.5))

x = [1, 2, 3]  # shihtl> epoch

for num_of_layers in [2, 3, 4]:
    y = avg_loss_data[f"LSTM_{num_of_layers}_{1e-3}"]
    plt.plot(x, y, label=num_of_layers, marker="o")

plt.legend(loc="center right")
plt.suptitle("Compare Average Training Loss Under Different Layer Numbers", fontsize=12)
plt.title("layer_type=LSTM, learning_rate=1e-3, vocab_dict=ASCII order", fontsize=8)
plt.xlabel("Epoch")
plt.xticks(x)
plt.ylabel("Avg. Training Loss")
plt.savefig(
    "Compare Average Training Loss Under Different Layer Numbers.png".replace(" ", "_"),
    bbox_inches="tight",
)

# shihtl> 比較不同建立字典方式的 accuracy
plt.figure(figsize=(16 / 2.5, 9 / 2.5))

x = [1, 2, 3]  # shihtl> epoch

for method in ["increasing", "descending"]:
    y = accuracy_data[f"LSTM_3_1e-3_vocab_by_freq_{method}"]
    plt.plot(x, y, label="freq. " + method, marker="o")
y = accuracy_data["LSTM_3_1e-3"]
plt.plot(x, y, label="ASCII order", marker="o")

plt.legend()
plt.suptitle("Compare Validation Accuracy With Different Vocab Dictionary", fontsize=12)
plt.title("layer_type=LSTM, num_of_layer=3, learning_rate=1e-3", fontsize=8)
plt.xlabel("Epoch")
plt.xticks(x)
plt.ylabel("Validation Accuracy")
plt.ylim(0, 1)
plt.savefig(
    "Compare Validation Accuracy With Different Vocab Dictionary.png".replace(" ", "_"),
    bbox_inches="tight",
)

# shihtl> 比較不同建立字典方式的 loss
plt.figure(figsize=(16 / 2.5, 9 / 2.5))

x = [1, 2, 3]  # shihtl> epoch

for method in ["increasing", "descending"]:
    y = avg_loss_data[f"LSTM_3_{1e-3}_vocab_by_freq_{method}"]
    plt.plot(x, y, label="freq. " + method, marker="o")
y = avg_loss_data[f"LSTM_3_{1e-3}"]
plt.plot(x, y, label="ASCII order", marker="o")

plt.legend(loc="center right")
plt.suptitle(
    "Compare Average Training Loss With Different Vocab Dictionary", fontsize=12
)
plt.title("layer_type=LSTM, num_of_layer=3, learning_rate=1e-3", fontsize=8)
plt.xlabel("Epoch")
plt.xticks(x)
plt.ylabel("Avg. Training Loss")
plt.savefig(
    "Compare Average Training Loss With Different Vocab Dictionary.png".replace(
        " ", "_"
    ),
    bbox_inches="tight",
)
