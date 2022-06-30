import torch
import torch.nn as nn
from itertools import islice

import wandb

from globals import label_vocab, ner_label_length

def perf(model, loader, epoch=1, dataset="eval"):
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

  eval_loss, eval_accuracy = total_loss / num_loss, correct.item() / num_perf
  wandb.log({f"{model.__class__.__name__}_{dataset}_loss": eval_loss, f"{model.__class__.__name__}_{dataset}_accuracy": eval_accuracy, f"{model.__class__.__name__}_{dataset}_epoch": epoch})

  return eval_loss, eval_accuracy

# Precision: percentage of named entity guesses that are exact matches
# Recall: Percentage of named entities found
def ner_perf(model, loader, epoch = 1, dataset="eval"):
  # we calculate the F1 score to evaluate performance
  criterion = nn.CrossEntropyLoss()
  # sets the model to evaluation mode
  model.eval()
  total_loss = num_loss = precision_correct = precision_num = recalled = entity_num = 0
  for x, y in loader:
    # disable gradient calculation
    with torch.no_grad():
      # perform inference and compute loss
      y_scores = model(x)
      # requires tensors of shape (num-instances, num-labels) and (num-instances)
      loss = criterion(y_scores.view(-1, ner_label_length), y.view(-1))

      # gather loss statistics
      total_loss += loss.item()
      num_loss += 1

      # gather accuracy statistics
      # compute highest-scoring tag
      y_pred = torch.max(y_scores, 2)[1]
      # ignore <pad> tags
      for i, labels in enumerate(y):
        for j, label in enumerate(labels):
          if y_pred[i][j] != 0 and y_pred[i][j] != 9:
            precision_num += 1
            if y_pred[i][j] == label:
              precision_correct += 1

          if label != 9 and label != 0:
            entity_num += 1
            if y_pred[i][j] != 0 and y_pred[i][j] != 9:
              recalled += 1

  eval_loss, eval_precision, eval_recall = total_loss / num_loss, precision_correct / precision_num, recalled / entity_num 
  wandb.log({f"{model.__class__.__name__}_{dataset}_loss": eval_loss, f"{model.__class__.__name__}_{dataset}_precision": eval_precision, f"{model.__class__.__name__}_{dataset}_recall": eval_recall, f"{model.__class__.__name__}_{dataset}_epoch": epoch})

  return eval_loss, eval_precision, eval_recall