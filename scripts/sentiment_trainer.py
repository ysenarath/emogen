import os
import shutil

import numpy as np
import torch
from collators import DataCollatorWithPadding
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from utils import config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

seed = 42

sigmoid = torch.nn.Sigmoid()
softmax = torch.nn.Softmax(dim=1)

base_model_name = "FacebookAI/roberta-base"

dataset_name = "stanfordnlp/sentiment140"

sentiment2label = {0: "negative", 2: "neutral", 4: "positive"}
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {0: "negative", 1: "neutral", 2: "positive"}


def map_to_label_id(example):
    sentiment = example["sentiment"]
    example["labels"] = label2id[sentiment2label[sentiment]]
    return example


# ['text', 'date', 'user', 'sentiment', 'query', 'labels']
# drop columns: date, user, query, sentiment
dsets = load_dataset(dataset_name, trust_remote_code=True).map(
    map_to_label_id, remove_columns=["date", "user", "query", "sentiment"]
)
# original_test_dataset = dsets["test"]
valid_dataset = (
    dsets["train"].shuffle(seed=42).train_test_split(test_size=1000, seed=42)
)
original_train_dataset = valid_dataset["train"]
original_valid_dataset = valid_dataset["test"]

num_labels = len(label2id)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)


def one_hot_encode_labels(example, num_labels):
    # for label in example["labels"]:
    #     one_hot_labels[int(label)] = 1.0
    example["labels"] = int(example["labels"])
    return example


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


# train_dataset = original_train_dataset.shuffle(seed=42).select(range(1000))
# valid_dataset = original_valid_dataset.shuffle(seed=42).select(range(1000))
train_dataset = original_train_dataset
valid_dataset = original_valid_dataset

train_dataset = train_dataset.map(
    one_hot_encode_labels, fn_kwargs={"num_labels": num_labels}
)
valid_dataset = valid_dataset.map(
    one_hot_encode_labels, fn_kwargs={"num_labels": num_labels}
)

train_dataset = train_dataset.map(tokenize_function, batched=True)
valid_dataset = valid_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch")
valid_dataset.set_format(type="torch")
train_dataset.set_format(
    type="torch", columns=["labels"], dtype=torch.float, output_all_columns=True
)
valid_dataset.set_format(
    type="torch", columns=["labels"], dtype=torch.float, output_all_columns=True
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt",
)


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, y_true = eval_pred
    if y_true.ndim == 2:  # noqa: PLR2004
        y_true = y_true.argmax(axis=1)
    cols = np.unique(y_true)
    cols.sort()
    # if binary classification, select the positive class logits
    #   (assume higher label index is positive)
    logits = logits[:, cols[1]] if len(cols) == 2 else logits[:, cols]  # noqa: PLR2004
    # replace y_true values with corrosponding index in cols
    col2idx = {col: idx for idx, col in enumerate(cols)}
    y_true = np.vectorize(col2idx.get)(y_true)

    if logits.ndim == 1:
        y_scores = sigmoid(torch.Tensor(logits)).numpy()
        y_pred = (y_scores > 0.5).astype(int)  # noqa: PLR2004
    else:
        y_scores = softmax(torch.Tensor(logits)).numpy()
        y_pred = np.argmax(y_scores, axis=1)

    scores = {}
    scores["accuracy"] = accuracy_score(y_true, y_pred)

    scores["roc_auc"] = roc_auc_score(
        y_true, y_scores, multi_class="ovr", average="macro"
    )
    for metric_name, compute_metric in {
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
    }.items():
        scores[metric_name] = compute_metric(y_true, y_pred, zero_division="warn")
    return scores


train_dir = config["train_dir"]

if os.path.exists(train_dir):
    shutil.rmtree(train_dir)

hub_model_id = "roberta-base-sentiment140"

training_args = TrainingArguments(
    output_dir=config["train_dir"],
    eval_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    # per_device_train_batch_size=32,
    # gradient_accumulation_steps=2, # Effective batch size = 32 * 2 = 64
    num_train_epochs=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    push_to_hub=True,
    hub_model_id=hub_model_id,
    # Mixed precision training
    # fp16=True,
    # fp16_opt_level="O2",
    # Additional optimizations
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    # gradient_checkpointing=True,
    seed=seed,
    report_to="none",
)


def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        problem_type="single_label_classification",
        id2label=id2label,
        label2id=label2id,
    )
    return model


model = model_init()

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001,
        )
    ],
)


trainer.train()

trainer.push_to_hub()

tokenizer.push_to_hub(hub_model_id)
