from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# Load the cleaned dataset
dataset = load_dataset("csv", data_files={"train": "./data/debate_toxicity_dataset.csv"})

# Map textual labels to numeric IDs
label2id = {"neutral": 0, "warning": 1, "toxic": 2}
id2label = {v: k for k, v in label2id.items()}

def encode_labels(example):
    example["label"] = label2id[example["label"]]
    return example

dataset = dataset.map(encode_labels)

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3, id2label=id2label, label2id=label2id
)

# Tokenize dataset
def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=64)

encoded_dataset = dataset.map(preprocess, batched=True)

# ✅ Fixed: use eval_strategy instead of evaluation_strategy
args = TrainingArguments(
    output_dir="./debate_toxicity_model",
    eval_strategy="no",          # <--- fixed line
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_dataset["train"],
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./debate_toxicity_model")
tokenizer.save_pretrained("./debate_toxicity_model")

print("✅ Fine-tuned model saved at ./debate_toxicity_model")
