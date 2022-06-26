import wandb
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from itertools import islice
from globals import label_vocab, lr, ner_label_length
from utils.accuracy import perf, ner_perf

def fit(model, epochs, train_loader, eval_loader):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  for epoch in range(epochs):
    model.train()
    total_loss = num = 0
    total_count = len(train_loader)

    for x, y in train_loader:
      # set all gradients to zero
      optimizer.zero_grad()
      y_scores = model(x)
      loss = criterion(y_scores.view(-1, len(label_vocab)), y.view(-1))
      # compute gradients though computation graph
      loss.backward() 
      # performs a single optimization step
      optimizer.step()
      total_loss += loss.item()
      num += 1
      wandb.log({f"{model.__class__.__name__}_train_loss": loss, f"{model.__class__.__name__}_epoch": epoch, f"{model.__class__.__name__}_progress": num / total_count})
    print(f"[Epoch {1+epoch}]\nTraining loss {total_loss / num}")
    eval_loss, eval_accuracy = perf(model, eval_loader)
    print(f"Eval loss {eval_loss} Eval accuracy {eval_accuracy}")

def ner_fit(model, epochs, train_loader, eval_loader):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  for epoch in range(epochs):
    model.train()
    total_loss = num = 0
    total_count = len(train_loader)

    for x, y in train_loader:
      # set all gradients to zero
      optimizer.zero_grad()
      y_scores = model(x)
      loss = criterion(y_scores.view(-1, ner_label_length), y.view(-1))
      # compute gradients though computation graph
      loss.backward() 
      # performs a single optimization step
      optimizer.step()
      total_loss += loss.item()
      num += 1
      wandb.log({f"{model.__class__.__name__}_train_loss": loss, f"{model.__class__.__name__}_epoch": epoch, f"{model.__class__.__name__}_progress": num / total_count})
    print(f"[Epoch {1+epoch}]\nTraining loss {total_loss / num}")
    eval_loss, eval_accuracy = ner_perf(model, eval_loader)
    print(f"Eval loss {eval_loss} Eval accuracy {eval_accuracy}")