from __future__ import annotations

from datasets import load_dataset

dataset_name = "stanfordnlp/sentiment140"
model_name = "ysenarath/roberta-base-sentiment140"

df = load_dataset(dataset_name, split="train").to_pandas()

print(df["sentiment"].value_counts())
