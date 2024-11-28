import pandas as pd

# Load the dataset
df_train = pd.read_csv("./kaggle/train.csv")
df_test = pd.read_csv("./kaggle/test.csv")
df_persuade_train = pd.read_csv(
    "./persuade_corpus_2.0/persuade_2.0_human_scores_demo_id_github.csv"
)

print("Train dataset shape: ", df_train.shape)
print("Test dataset shape: ", df_test.shape)
print("Persuade Train dataset shape: ", df_persuade_train.shape)

print(df_train.columns)
print(df_test.columns)
print(df_persuade_train.columns)

# 移除不需要的欄位，有需要可以註解掉
drop_columns = [
    "word_count",
    "prompt_name",
    "task",
    "assignment",
    "source_text",
    "gender",
    "grade_level",
    "ell_status",
    "race_ethnicity",
    "economically_disadvantaged",
    "student_disability_status",
]

df_persuade_train = df_persuade_train.drop(columns=drop_columns)
df_persuade_train = df_persuade_train.rename(columns={"essay_id_comp": "essay_id"})
df_persuade_train = df_persuade_train.rename(columns={"holistic_essay_score": "score"})

df_only_kaggle_train = pd.DataFrame(columns=["essay_id", "full_text", "score"])
df_both_kaggle_persuade_train = pd.DataFrame(columns=["essay_id", "full_text", "score"])
df_only_persuade_train = pd.DataFrame(columns=["essay_id", "full_text", "score"])

# 在 train 中，不在 persuade_train 中的數據
only_kaggle_train = ~df_train["full_text"].isin(df_persuade_train["full_text"])
df_only_kaggle_train = df_train.loc[only_kaggle_train]
df_only_kaggle_train.to_csv(
    "./separated/kaggle_train.csv", encoding="utf-8", index=False
)
print("Only Kaggle Train dataset shape: ", df_only_kaggle_train.shape)

# 在 train 中，同時在 persuade_train 中的數據
df_both_kaggle_persuade_train = df_train.loc[~only_kaggle_train]
df_both_kaggle_persuade_train.to_csv(
    "./separated/kaggle_persuade_train.csv", encoding="utf-8", index=False
)
print(
    "Both Kaggle and Persuade Train dataset shape: ",
    df_both_kaggle_persuade_train.shape,
)

# 在 persuade_train 中，不在 train 中的數據
only_persuade_train = ~df_persuade_train["full_text"].isin(df_train["full_text"])
df_only_persuade_train = df_persuade_train.loc[only_persuade_train]
df_only_persuade_train.to_csv(
    "./separated/persuade_train.csv", encoding="utf-8", index=False
)
print("Only Persuade Train dataset shape: ", df_only_persuade_train.shape)
