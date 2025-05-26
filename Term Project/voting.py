# import os
# import re
# import random
# import subprocess
import warnings

# import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
import torch
from typing import Literal
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

# from transformers import DataCollatorWithPadding
# from transformers import TextClassificationPipeline
# from transformers.pipelines.pt_utils import KeyDataset
# from datasets import Dataset
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.metrics import cohen_kappa_score
# from IPython.display import FileLink, display

warnings.simplefilter("ignore")


class VotingModel(torch.nn.Module):
    def __init__(
        self,
        model_paths: list,
        tokenizer_path: str = None,
        voting_mode: Literal["mean", "linear"] = "mean",
        linear_layer_num: int = 0,  # if voting_mode is "linear"
    ) -> None:
        super(VotingModel, self).__init__()

        self.models = [
            AutoModelForSequenceClassification.from_pretrained(path, num_labels=1)
            for path in model_paths
        ]
        self.tmp_linear = torch.nn.Linear(6, 1).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = (
            AutoTokenizer.from_pretrained(tokenizer_path)
            if tokenizer_path
            else AutoTokenizer.from_pretrained("microsoft/deberta-base")
        )

        assert (
            linear_layer_num >= 0
        ), "linear_layer_num must be greater than or equal to 0"
        self.voting_mode = "mean" if linear_layer_num == 0 else "linear"

        num_of_models = len(self.models)
        if self.voting_mode == "linear":
            self.linear_list = [
                torch.nn.Sequential(
                    torch.nn.Linear(num_of_models, num_of_models), torch.nn.ReLU()
                )
                for _ in range(linear_layer_num - 1)
            ] + [torch.nn.Linear(num_of_models, 1)]
            self.linears = torch.nn.Sequential(*self.linear_list)
            print(self.linears)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # initialize settings of models
        for model in self.models:
            model.to(self.device)
            model.eval()
            # model.parameters().requires_grad = False  # freeze all pre-trained layers

        if self.voting_mode == "linear":
            self.linears.to(self.device)

    def forward(self, X):
        X = self.tokenizer(X, return_tensors="pt", padding=True, truncation=True)
        X = {k: v.to(self.device) for k, v in X.items()}

        with torch.no_grad():
            print([model(**X) for model in self.models])
            # y = torch.stack(
            #     # [self.tmp_linear(model(**X).logits) for model in self.models]
            #     [self.tmp_linear(model(**X).logits) for model in self.models]
            # )
            y = 0

        if self.voting_mode == "mean":
            y = y.mean(dim=0)
        elif self.voting_mode == "linear":
            y = y.squeeze(2).transpose(0, 1)
            # print(y.shape)
            y = self.linears(y)

        return y


def main():
    voting_model = VotingModel(
        [
            "./model/alef-b-220-regv1-pc2-deberta-v3-small-skf-s5-e4-pytorch-v1-v1/output_v1/checkpoint-17396",
        ],
        tokenizer_path="./model/alef-b-220-regv1-pc2-deberta-v3-small-skf-s5-e4-pytorch-v1-v1/output_v1/checkpoint-17396-tokenizer",
    )

    test_df = pd.read_csv("./dataset/kaggle/test.csv")
    test_scores = []
    res = voting_model(test_df["full_text"].tolist())
    print(res)


if __name__ == "__main__":
    main()
