# %% [markdown]
# # LSTM-arithmetic
#
# ## Dataset
# - [Arithmetic dataset](https://drive.google.com/file/d/1cMuL3hF9jefka9RyF4gEBIGGeFGZYHE-/view?usp=sharing)

# %%
# ! pip install seaborn
# ! pip install opencc
# ! pip install -U scikit-learn

import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.utils.rnn
import torch.utils.data
import matplotlib.pyplot as plt
import seaborn as sns
import opencc
import os
from sklearn.model_selection import train_test_split

data_path = "./data"

# %%
df_train = pd.read_csv(os.path.join(data_path, "arithmetic_train.csv"))
df_eval = pd.read_csv(os.path.join(data_path, "arithmetic_eval.csv"))
df_train.head()

# shihtl> tmp reducing training size
# df_train = df_train[:12800]
# df_eval = df_eval[:1]

# %%
# transform the input data to string
df_train["tgt"] = df_train["tgt"].apply(lambda x: str(x))
df_train["src"] = df_train["src"].add(df_train["tgt"])
df_train["len"] = df_train["src"].apply(lambda x: len(x))

df_eval["tgt"] = df_eval["tgt"].apply(lambda x: str(x))
df_eval["src"] = df_eval["src"].add(df_eval["tgt"])
df_eval["len"] = df_eval["src"].apply(lambda x: len(x))

# %%
df_train.head()

# %% [markdown]
# # Build Dictionary
#  - The model cannot perform calculations directly with plain text.
#  - Convert all text (numbers/symbols) into numerical representations.
#  - Special tokens
#     - '&lt;pad&gt;'
#         - Each sentence within a batch may have different lengths.
#         - The length is padded with '&lt;pad&gt;' to match the longest sentence in the batch.
#     - '&lt;eos&gt;'
#         - Specifies the end of the generated sequence.
#         - Without '&lt;eos&gt;', the model will not know when to stop generating.

# %%
char_to_id = {
    "<pad>": 0,
    "<eos>": 1,
}

id_to_char = {}


# write your code here
# Build a dictionary and give every token in the train dataset an id
# The dictionary should contain <eos> and <pad>
# char_to_id is to conver charactors to ids, while id_to_char is the opposite

# shihtl> 這裡遍歷所有的 lines in df_train["src"] 中所有的 char，但每次都遍歷有點多餘，目前不會有新的 char，所以執行一次之後改成直接指定 char_to_id
# shihtl> method 1.0
# unique_chars = []
# for line in df_train["src"]:
#     for char in line:
#         if char not in unique_chars:
#             unique_chars.append(char)
# unique_chars.sort()

# for idx, val in enumerate(unique_chars, start=2):
#     char_to_id[val] = idx

# shihtl> method 1.1
char_to_id = {
    "<pad>": 0,
    "<eos>": 1,
    "(": 2,
    ")": 3,
    "*": 4,
    "+": 5,
    "-": 6,
    "0": 7,
    "1": 8,
    "2": 9,
    "3": 10,
    "4": 11,
    "5": 12,
    "6": 13,
    "7": 14,
    "8": 15,
    "9": 16,
    "=": 17,
}

# shihtl> gen id_to_char
for key, val in char_to_id.items():
    id_to_char[val] = key


vocab_size = len(char_to_id)

print("Vocab size: {}".format(vocab_size))
print(char_to_id)
print(id_to_char)

# %% [markdown]
# # Data Preprocessing
#  - The data is processed into the format required for the model's input and output.
#  - Example: 1+2-3=0
#      - Model input: 1 + 2 - 3 = 0
#      - Model output: / / / / / 0 &lt;eos&gt;  (the '/' can be replaced with &lt;pad&gt;)
#      - The key for the model's output is that the model does not need to predict the next character of the previous part. What matters is that once the model sees '=', it should start generating the answer, which is '0'. After generating the answer, it should also generate&lt;eos&gt;
#


# %%
# Write your code here
def zero_before_eq_sign(row: list):
    """shihtl> 把第一個等號之前的項次，都變成 encoded <pad>"""
    idx = row.index(char_to_id["="])

    return [char_to_id["<pad>"]] * (idx + 1) + row[idx + 1 :]


for df in [df_train, df_eval]:
    df["char_id_list"] = df["src"].apply(
        lambda row: [char_to_id[char] for char in row] + [char_to_id["<eos>"]]
    )
    # df["char_id_list"] = df["src"].apply(lambda row: [char_to_id[char] for char in row])
    df["label_id_list"] = df["char_id_list"].apply(zero_before_eq_sign)


print(df_train.head())
print(df_eval.head())

# %% [markdown]
# # Hyper Parameters
#
# |Hyperparameter|Meaning|Value|
# |-|-|-|
# |`batch_size`|Number of data samples in a single batch|64|
# |`epochs`|Total number of epochs to train|10|
# |`embed_dim`|Dimension of the word embeddings|256|
# |`hidden_dim`|Dimension of the hidden state in each timestep of the LSTM|256|
# |`lr`|Learning Rate|0.001|
# |`grad_clip`|To prevent gradient explosion in RNNs, restrict the gradient range|1|

# %%
batch_size = 128
epochs = 1
embed_dim = 256
hidden_dim = 256
lr = 0.00001
grad_clip = 1

# %% [markdown]
# # Data Batching
# - Use `torch.utils.data.Dataset` to create a data generation tool called  `dataset`.
# - The, use `torch.utils.data.DataLoader` to randomly sample from the `dataset` and group the samples into batches.


# %%
class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        # return the amount of data
        return len(self.sequences)

    def __getitem__(self, index):
        # Extract the input data x and the ground truth y from the data
        x = self.sequences["char_id_list"][index]
        y = self.sequences["label_id_list"][index]
        return x, y


# collate function, used to build dataloader
def collate_fn(batch):
    batch_x = [torch.tensor(data[0]) for data in batch]
    batch_y = [torch.tensor(data[1]) for data in batch]
    batch_x_lens = torch.LongTensor([len(x) for x in batch_x])
    batch_y_lens = torch.LongTensor([len(y) for y in batch_y])

    # Pad the input sequence
    pad_batch_x = torch.nn.utils.rnn.pad_sequence(
        batch_x, batch_first=True, padding_value=char_to_id["<pad>"]
    )

    pad_batch_y = torch.nn.utils.rnn.pad_sequence(
        batch_y, batch_first=True, padding_value=char_to_id["<pad>"]
    )

    return pad_batch_x, pad_batch_y, batch_x_lens, batch_y_lens


# %%
ds_train = Dataset(df_train[["char_id_list", "label_id_list"]])
ds_eval = Dataset(df_eval[["char_id_list", "label_id_list"]])

# %%
for i in range(10):
    print(ds_train[i])

# %%
# Build dataloader of train set and eval set, collate_fn is the collate function
dl_train = torch.utils.data.DataLoader(
    ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
dl_eval = torch.utils.data.DataLoader(
    ds_eval, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

# %% [markdown]
# # Model Design
#
# ## Execution Flow
# 1. Convert all characters in the sentence into embeddings.
# 2. Pass the embeddings through an LSTM sequentially.
# 3. The output of the LSTM is passed into another LSTM, and additional layers can be added.
# 4. The output from all time steps of the final LSTM is passed through a Fully Connected layer.
# 5. The character corresponding to the maximum value across all output dimensions is selected as the next character.
#
# ## Loss Function
# Since this is a classification task, Cross Entropy is used as the loss function.
#
# ## Gradient Update
# Adam algorithm is used for gradient updates.


# %%
class CharRNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(CharRNN, self).__init__()

        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=char_to_id["<pad>"],
        )

        self.rnn_layer1 = torch.nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim, batch_first=True
        )

        # self.rnn_layer2 = torch.nn.LSTM(
        #     input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True
        # )

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_dim, out_features=vocab_size),
        )

    def forward(self, batch_x, batch_x_lens):
        return self.encoder(batch_x, batch_x_lens)

    # The forward pass of the model
    def encoder(self, batch_x, batch_x_lens):
        batch_x = self.embedding(batch_x)

        batch_x = torch.nn.utils.rnn.pack_padded_sequence(
            batch_x, batch_x_lens, batch_first=True, enforce_sorted=False
        )

        batch_x, _ = self.rnn_layer1(batch_x)
        # batch_x, _ = self.rnn_layer2(batch_x)

        batch_x, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_x, batch_first=True)

        batch_x = self.linear(batch_x)

        return batch_x

    def generator(self, start_char, max_len=200):
        print(start_char)

        char_list = [char_to_id[c] for c in start_char]

        next_char = None

        while len(char_list) < max_len:
            # Write your code here
            # Pack the char_list to tensor
            x = torch.tensor(char_list)
            x = self.embedding(x)
            x, _ = self.rnn_layer1(x)
            # x, _ = self.rnn_layer2(x)
            y = self.linear(x)

            print(y.shape)

            y = y[0, -1, :]

            next_char = torch.argmax(y).item()
            # print(id_to_char[next_char])

            if next_char == char_to_id["<eos>"]:
                break

            char_list.append(next_char)

        return [id_to_char[ch_id] for ch_id in char_list]


# %%
torch.manual_seed(2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CharRNN(vocab_size, embed_dim, hidden_dim)

# %%
criterion = torch.nn.CrossEntropyLoss(ignore_index=char_to_id["<pad>"])
# criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# %% [markdown]
# # Training
# 1. The outer `for` loop controls the `epoch`
#     1. The inner `for` loop uses `data_loader` to retrieve batches.
#         1. Pass the batch to the `model` for training.
#         2. Compare the predicted results `batch_pred_y` with the true labels `batch_y` using Cross Entropy to calculate the loss `loss`
#         3. Use `loss.backward` to automatically compute the gradients.
#         4. Use `torch.nn.utils.clip_grad_value_` to limit the gradient values between `-grad_clip` &lt; and &lt; `grad_clip`.
#         5. Use `optimizer.step()` to update the model (backpropagation).
# 2.  After every `1000` batches, output the current loss to monitor whether it is converging.


# %%
def two_tensor_are_same(pred, ans):
    """比較 pred 的數值有沒有跟 ans 非 0 部分相同，如果 ans 是 0 就忽略"""

    for num_pred, num_ans in zip(pred, ans):
        if num_ans == 0:
            continue
        elif num_pred != num_ans:
            return False

    return True


# %%
from tqdm import tqdm
from copy import deepcopy

torch.set_printoptions(threshold=np.inf)

model = model.to(device)
i = 0
for epoch in range(1, epochs + 1):
    model.train()
    # The process bar
    bar = tqdm(dl_train, desc=f"Train epoch {epoch}")
    for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:
        # Write your code here
        # Clear the gradient
        optimizer.zero_grad()

        batch_pred_y = model(batch_x.to(device), batch_x_lens)
        # batch_pred_y = batch_pred_y.view(-1, vocab_size)
        # batch_y = batch_y.view(-1)
        batch_pred_y = batch_pred_y.transpose(1, 2)

        # print(batch_y.shape)
        # print(batch_pred_y.shape)
        # print(batch_y[0])
        # # print(batch_pred_y[0])
        # print(batch_pred_y[0].T.argmax(dim=1))
        # # print(batch_pred_y[0].argmax(dim=1))
        # input()

        # Write your code here
        # Input the prediction and ground truths to loss function
        loss = criterion(
            batch_pred_y, batch_y.to(device)
        )  # shihtl> This line of code was generated by GPT-4o.
        # input()
        # Back propagation
        loss.backward()

        torch.nn.utils.clip_grad_value_(
            model.parameters(), grad_clip
        )  # gradient clipping

        # Write your code here
        # Optimize parameters in the model
        optimizer.step()

        i += 1
        if i % 50 == 0:
            # print(loss.item())
            # print(batch_y[0])
            # print(batch_pred_y[0].argmax(dim=1))
            bar.set_postfix(loss=loss.item())

    # Evaluate your model
    model.eval()
    bar = tqdm(dl_eval, desc=f"Validation epoch {epoch}")
    matched = 0
    total = 0
    for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:

        # predictions = # Write your code here. Input the batch_x to the model and generate the predictions
        predictions = model(batch_x.to(device), batch_x_lens)
        # print(predictions)
        # print(predictions.argmax(dim=2))
        # input()

        # Write your code here.
        # Check whether the prediction match the ground truths
        # for pred, ans in zip(predictions, batch_y):
        #     if two_tensor_are_same(pred, ans):
        #         matched += 1

        # input()
        # match_res = (predictions == batch_y.to(device)).tolist()
        # matched += match_res.count(True)
        total += len(batch_y_lens)
        # Compute exact match (EM) on the eval dataset
        EM = matched / total

    print(matched / total)

# %% [markdown]
# # Generation
# Use `model.generator` and provide an initial character to automatically generate a sequence.

# %%
model = model.to("cpu")
# print(model.generator('1+1='))
print("".join(model.generator("3+9=")))
