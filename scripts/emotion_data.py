import os

import datasets

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dataset_name, dataset_config_name = "go_emotions", "simplified"
dataset_dict = datasets.load_dataset(dataset_name, dataset_config_name)

dataset_config = datasets.get_dataset_config_info(dataset_name, dataset_config_name)

classes = dataset_config.features["labels"].feature.names

emotions2sentiment = {
    "admiration": "positive",
    "amusement": "positive",
    "anger": "negative",
    "annoyance": "negative",
    "approval": "positive",
    "caring": "positive",
    "confusion": "neutral",
    "curiosity": "neutral",
    "desire": "positive",
    "disappointment": "negative",
    "disapproval": "negative",
    "disgust": "negative",
    "embarrassment": "negative",
    "excitement": "positive",
    "fear": "negative",
    "gratitude": "positive",
    "grief": "negative",
    "joy": "positive",
    "love": "positive",
    "nervousness": "negative",
    "optimism": "positive",
    "pride": "positive",
    "realization": "neutral",
    "relief": "positive",
    "remorse": "negative",
    "sadness": "negative",
    "surprise": "neutral",
    "neutral": "neutral",
}


# assert every emotion has a sentiment
assert set(emotions2sentiment.keys()) == set(classes)
