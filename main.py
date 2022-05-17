from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch

def Tokenize(data):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    a = data["sentence"]
    return tokenizer(a, padding="max_length", truncation=True)

def main():
    dataset = load_dataset("aakanksha/udpos")
    print(dataset["train"][0])
    tokenized_dataset = dataset.map(Tokenize, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence"])
    tokenized_dataset = tokenized_dataset.rename_column("tags", "labels")
    tokenized_dataset.set_format("torch")
    small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))

    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    device = device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))
    model.train()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    
    metric = load_metric("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    
    metric.compute()

if __name__ == '__main__':
    main()