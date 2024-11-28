# Rank 1st
1. Data analysis + experiments to find "new" data scoring patterns and get CV-LB-correlation.
    - 想辦法分析 validation set 和 leaderboard 分數的關係，根據這個去優化 algo
2. Train Deberta ensemble reflecting these scores.
    - DeBERTa 綜整這些分數的關係（base model: Deberta large, base）
3. Two rounds of pseudo labelling to get "old" data with "new" scores.
    - 用兩輪的 pseudo label 來得到舊資料的新分數
    - 這是因為他發現資料集裡面有 ~13k 的舊資料，和 4.5k 之前沒看過的東西（所有有兩個 dataset 合併起來的？）
4. Transform float predictions via thresholding to ints.
    - 因為 predict 出來的是 float 分數，用 thresholding 來判斷最終要分類到哪一個分數
5. Ensemble without overfitting.
    - 盡量不要 overfitting，但怎麼感覺在講廢話
    - ensemble 貌似是指產生最後的預測分數 label

其他：
- stratified 5-fold split using prompt_id+score as labels
- fine-tune 資料集只有使用 4.5k，盡量六種分數都有。他有發現不同 seed 得到的分數差異很大，特別是用 threshold 來將 float 轉成 int 分數，於是他用三個 seed 來平均
- 因為要用三個 seed 算平均所以訓練時間和成本很大，他發現重新訓練 fine-tune phase 的 model 就好，model folder 他大概用了 2 TB
- 因為 score label 的分佈部分很少，這個導致的 error 要處理掉。`true_label==2` 常常被錯誤分類到 `3`
- `"\n"` 和 `" "` 不應該被忽略，應該要 encode

# Rank 2nd
1. 訓練了五個模型來投票
2. 有明確提到這個資料集是兩個資料集合併而成的
3. 4 folds
4. 用舊資料 train base model，用新資料 fine-tune
5. 有完整的 code 在 GitHub

# Rank 3rd
1. 有完整的 code 在 Kaggle

# Rank 4th
1. 用 mini batch 來減少 batch padding 的需求
2. 針對效率部分對 DeBERTa 優化架構
3. 有完整的 code 在 Kaggle