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
  wandb.log({f"{dataset}_loss": eval_loss, f"{dataset}_accuracy": eval_accuracy, f"{dataset}_epoch": epoch})

  return eval_loss, eval_accuracy

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

      y_pred = torch.max(y_scores, 2)[1]

      for i, labels in enumerate(y):
        for j, label in enumerate(labels):
          # Precision: percentage of named entity guesses that are exact matches
          if is_entity(y_pred[i][j]):
            precision_num += 1
            if y_pred[i][j] == label:
              precision_correct += 1
          # Recall: Percentage of named entities found
          if is_entity(label):
            entity_num += 1
            if is_entity(y_pred[i][j]):
              recalled += 1

  eval_loss, eval_precision, eval_recall = total_loss / num_loss, _division(precision_correct, precision_num), _division(recalled, entity_num)
  wandb.log({f"{dataset}_loss": eval_loss, f"{dataset}_precision": eval_precision, f"{dataset}_recall": eval_recall, f"{dataset}_epoch": epoch})

  return eval_loss, eval_precision, eval_recall

def _division(n, d):
  return n / d if d else 0

def is_entity(y):
  # 9 is a pad tag, 0 is a non-entity
  return y != 9 and y != 0