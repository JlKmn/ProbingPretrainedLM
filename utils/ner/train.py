import imp
import wandb
import torch.optim as optim
import torch.nn as nn

from globals import ner_label_length
from utils.ner.accuracy import perf

def fit(model, train_loader, eval_loader, params):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=params.lr)
  for epoch in range(params.epochs):
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
      wandb.log({f"train_loss": loss})
    print(f"[Epoch {1+epoch}]\nTraining loss {total_loss / num}")
    eval_loss, eval_precision, eval_recall = perf(model, eval_loader, epoch=epoch)
    print(f"Eval loss {eval_loss} Eval precision {eval_precision} Eval recall {eval_recall}")