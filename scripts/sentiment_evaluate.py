from __future__ import annotations

import pandas as pd
from datasets import load_dataset
from rich.progress import track
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset
from transformers import pipeline


class Pipeline(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __getitem__(self, i: int | list[int] | slice) -> str | list[str]:
        text = self.df["text"].iloc[i]
        if isinstance(text, pd.Series):
            return text.tolist()
        return text

    def __len__(self) -> int:
        return self.df.shape[0]

    def pipe(self, func):
        df = pd.DataFrame(track(func(self), total=len(self))).rename(
            columns={"label": "pred_label", "score": "pred_score"}
        )
        return Pipeline(pd.concat([self.df, df], axis=1))

    def results(self):
        return self.df


dataset_name = "stanfordnlp/sentiment140"
model_name = "ysenarath/roberta-base-sentiment140"

df = load_dataset(dataset_name, split="test").to_pandas()
sentiment2label = {0: "negative", 2: "neutral", 4: "positive"}
df["label"] = df["sentiment"].map(sentiment2label)

# drop neutral sentiment
# df = df[df["label"] != "neutral"].reset_index(drop=True)

df = (
    Pipeline(df)
    .pipe(pipeline("text-classification", model=model_name, device_map="auto"))
    .results()
)

y_true = df["label"].map({"positive": 2, "neutral": 1, "negative": 0})
y_pred = df["pred_label"].map({"positive": 2, "neutral": 1, "negative": 0})

print("Accuracy:", accuracy_score(y_true, y_pred))
# print("ROC AUC:", roc_auc_score(y_true, y_score, average="macro", multi_class="ovr"))
print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
print("Macro Precision:", precision_score(y_true, y_pred, average="macro"))
print("Macro Recall:", recall_score(y_true, y_pred, average="macro"))
