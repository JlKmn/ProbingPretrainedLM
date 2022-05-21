from cProfile import label
import os
from datasets import load_dataset
import torch
import argparse
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments

args = None
OUTPUT_PATH = "model"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples[f"pos_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def main():
    print("Starting")
    dataset = load_dataset("xtreme", "udpos.English")
    example = dataset["train"][0]
    print(f"Example {example}")
    label_list = dataset["train"].features[f"pos_tags"].feature.names
    print(f"pos labels {label_list}")
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
    small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(500))
    small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(500))
    print(tokenized_dataset["train"][0])

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    model = None
    if os.path.exists(OUTPUT_PATH):
        model = torch.load(OUTPUT_PATH)
    else:
        model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_list))

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if args.action == "train":
        trainer.train()
    elif args.action == "eval":
        trainer.train()

    torch.save(model, OUTPUT_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate or train probing model')
    parser.add_argument('action', type=str, default="train", help='Action to perform')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    args = parser.parse_args()
    main()