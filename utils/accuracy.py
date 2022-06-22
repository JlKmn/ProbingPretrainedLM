import torch
import torch.nn as nn
from itertools import islice

import wandb

from globals import label_vocab

def perf(model, loader, dataset="eval"):
  criterion = nn.CrossEntropyLoss()
  # sets the model to evaluation mode
  model.eval()
  total_loss = correct = num_loss = num_perf = 0
  for x, y in loader:
    # disable gradient calculation
    with torch.no_grad():
      # perform inference and compute loss
      y_scores = model(x)
      # requires tensors of shape (num-instances, num-labels) and (num-instances)
      loss = criterion(y_scores.view(-1, len(label_vocab)), y.view(-1))

      # gather loss statistics
      total_loss += loss.item()
      num_loss += 1

      # gather accuracy statistics
      # compute highest-scoring tag
      y_pred = torch.max(y_scores, 2)[1]
      # ignore <pad> tags
      mask = (y != 0)
      # compute number of correct predictions
      correct += torch.sum((y_pred == y) * mask)
      num_perf += torch.sum(mask).item()
      l, a = total_loss / num_loss, correct.item() / num_perf 
      wandb.log({f"{model.__class__.__name__}_{dataset}_loss": l, f"{model.__class__.__name__}_{dataset}_accuracy": a})

  return total_loss / num_loss, correct.item() / num_perf 