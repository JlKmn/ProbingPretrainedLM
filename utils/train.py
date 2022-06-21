import wandb
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from itertools import islice
from globals import label_vocab, lr, BATCHES, EVAL_BATCHES
from utils.accuracy import perf

def fit(model, epochs, train_loader, eval_loader):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  for epoch in range(epochs):
    model.train()
    total_loss = num = 0
    for x, y in islice(train_loader, 0, BATCHES):
    #for x, y in train_loader:
      # set all gradients to zero
      optimizer.zero_grad()
      y_scores = model(x)
      loss = criterion(y_scores.view(-1, len(label_vocab)), y.view(-1))
      # compute gradients though computation graph
      loss.backward() 
      # performs a single optimization step
      optimizer.step()
      wandb.log({f"{model.__class__.__name__}__loss": loss, f"{model.__class__.__name__}_epoch": epoch})
      total_loss += loss.item()
      num += 1
    print(f"[Epoch {1+epoch}]\nTraining loss {total_loss / num}")
    eval_loss, eval_accuracy = perf(model, eval_loader, EVAL_BATCHES)
    wandb.log({f"{model.__class__.__name__}__eval_loss": eval_loss, f"{model.__class__.__name__}__eval_accuracy": eval_accuracy, f"{model.__class__.__name__}_epoch": epoch})
    print(f"Eval loss {eval_loss} Eval accuracy {eval_accuracy}")