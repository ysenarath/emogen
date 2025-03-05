from datasets import load_dataset

dataset_name = "stanfordnlp/sentiment140"

sentiment2label = {0: "negative", 2: "neutral", 4: "positive"}
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {0: "negative", 1: "neutral", 2: "positive"}


def map_to_label_id(example):
    sentiment = example["sentiment"]
    example["labels"] = label2id[sentiment2label[sentiment]]
    return example


datasets = load_dataset(dataset_name).map(map_to_label_id)
test_dataset = datasets["test"]
valid_dataset = (
    datasets["train"].shuffle(seed=42).train_test_split(test_size=1000, seed=42)
)
train_dataset = valid_dataset["train"]
valid_dataset = valid_dataset["test"]

print(train_dataset)
print(valid_dataset)
print(test_dataset)
