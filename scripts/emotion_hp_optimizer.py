import os
import shutil

import datasets
import numpy as np
import torch
import tqdm
from collators import DataCollatorWithPadding
from datasets import Dataset
from optuna import Trial
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
from utils import config, set_seed

sigmoid = torch.nn.Sigmoid()


os.environ["TOKENIZERS_PARALLELISM"] = "false"

seed = 0

set_seed(seed)


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


output_dir = config["optimize_dir"]

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

hub_model_id = "roberta-base-go_emotions"

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    # per_device_train_batch_size=32,
    # gradient_accumulation_steps=2, # Effective batch size = 32 * 2 = 64
    num_train_epochs=15,
    learning_rate=2e-5,
    # learning_rate=5e-5,
    # weight_decay=0.01,
    # warmup_ratio=0.1,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_macro_f1",
    push_to_hub=True,
    hub_model_id=hub_model_id,
    # Mixed precision training
    fp16=True,
    fp16_opt_level="O2",
    # Additional optimizations
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    # gradient_checkpointing=True,
    seed=seed,
)


def model_init(trial: Trial = None):
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )
    # Enable gradient checkpointing
    # model.gradient_checkpointing_enable()

    # model = BetterTransformer.transform(model)
    return model


model = model_init()


def stratified_multilabel_sample(dataset: Dataset, n_samples: int) -> Dataset:
    labels = train_dataset["labels"]
    n_classes = labels.shape[1]
    n_samples_per_class = n_samples // n_classes
    dsets = []
    datasets.disable_progress_bar()
    for i in tqdm.trange(n_classes):
        dset = dataset.filter(lambda x, j: x["labels"][j] == 1, fn_kwargs={"j": i})
        sample_size = dset.num_rows
        dsets.append(dset.select(range(min(sample_size, n_samples_per_class))))
    datasets.enable_progress_bar()
    return datasets.concatenate_datasets(dsets)


small_train_dataset = stratified_multilabel_sample(train_dataset, 10000)
small_valid_dataset = stratified_multilabel_sample(valid_dataset, 5000)


trainer = Trainer(
    model=model,
    model_init=model_init,
    args=training_args,
    data_collator=data_collator,
    train_dataset=small_train_dataset,
    eval_dataset=small_valid_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001,
        )
    ],
)


def optuna_hp_space(trial: Trial) -> dict[str, float]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [16, 32, 64]
        ),
    }


def compute_objective(metrics: dict[str, float]) -> float:
    return metrics["eval_macro_f1"]


best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=20,
    compute_objective=compute_objective,
)


print(best_trial.hyperparameters)
