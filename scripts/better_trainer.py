import os
import shutil

import datasets
import numpy as np
import torch
from collators import DataCollatorWithPadding
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
from transformers.models.roberta.modeling_roberta import (
    RobertaForSequenceClassification,
)
from utils import config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

seed = 42


sigmoid = torch.nn.Sigmoid()

base_model_name = "FacebookAI/roberta-base"


dataset_name, dataset_config_name = "go_emotions", "simplified"
dataset_dict = datasets.load_dataset(dataset_name, dataset_config_name)
original_train_dataset = dataset_dict["train"]
original_valid_dataset = dataset_dict["validation"]

dataset_config = datasets.get_dataset_config_info(dataset_name, dataset_config_name)

classes = dataset_config.features["labels"].feature.names

id2label = dict(enumerate(classes))
label2id = {label: i for i, label in id2label.items()}

num_labels = len(classes)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)


def one_hot_encode_labels(example, num_labels):
    one_hot_labels = np.zeros(num_labels).astype(np.float32)
    for label in example["labels"]:
        one_hot_labels[int(label)] = 1.0
    example["labels"] = one_hot_labels
    return example


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


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

    # drop columns where sum is 0
    cols = y_true.sum(axis=0) > 0
    logits = logits[:, cols]

    threshold = 0.5
    y_true = y_true[:, cols] >= threshold

    y_scores = sigmoid(torch.Tensor(logits)).numpy()
    y_pred = y_scores >= threshold

    scores = {}
    scores["accuracy"] = accuracy_score(y_true, y_pred)
    scores["roc_auc"] = roc_auc_score(
        y_true, y_scores, average="macro", multi_class="ovr"
    )

    for average in ["micro", "macro", "weighted"]:
        for metric_name, compute_metric in {
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
        }.items():
            scores[f"{average}_{metric_name}"] = compute_metric(
                y_true, y_pred, average=average, zero_division="warn"
            )
    return scores


train_dir = config["train_dir"]

if os.path.exists(train_dir):
    shutil.rmtree(train_dir)

hub_model_id = "roberta-base-go_emotions"

training_args = TrainingArguments(
    output_dir=config["train_dir"],
    eval_strategy="epoch",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    # per_device_train_batch_size=32,
    # gradient_accumulation_steps=2, # Effective batch size = 32 * 2 = 64
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_macro_f1",
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
)


def model_init(freeze_encoder: bool = True):
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    # model = BetterTransformer.transform(model)

    # Enable gradient checkpointing
    # model.gradient_checkpointing_enable()

    if freeze_encoder:
        if isinstance(model, RobertaForSequenceClassification):
            for param in model.roberta.parameters():
                param.requires_grad = False
        else:
            msg = f"model type {model.__class__.__name__} is not supported"
            raise TypeError(msg)

    return model


def unfreeze_encoder(model, top_layers: int = 1):
    if isinstance(model, RobertaForSequenceClassification):
        if len(model.roberta.encoder.layer) < top_layers + 1:
            msg = f"model has only {len(model.roberta.encoder.layer)} layers"
            raise ValueError(msg)
        for layer in model.roberta.encoder.layer[-top_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        freeze_embeddings = len(model.roberta.encoder.layer) == top_layers
        if freeze_embeddings:
            for param in model.roberta.embeddings.parameters():
                param.requires_grad = True
    else:
        msg = f"model type {model.__class__.__name__} is not supported"
        raise TypeError(msg)
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

top_layers = 1
while True:
    try:
        unfreeze_encoder(trainer.model, top_layers=top_layers)
    except ValueError:
        break

    trainer.train()

    top_layers += 1

    trainer.train()

trainer.push_to_hub()

tokenizer.push_to_hub(hub_model_id)
