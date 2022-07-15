from typing import List, Tuple, Set
import torch
import torch.nn as nn
import wandb

from src.globals import debug_print, ner_label_length

def perf(model, loader, epoch = 1, dataset="eval"):
  # we calculate the F1 score to evaluate performance
  criterion = nn.CrossEntropyLoss()
  # sets the model to evaluation mode
  model.eval()
  total_loss = num_loss = precision_sum = recall_sum = iterations = 0
  first = True
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
        s = ner_spans(labels)
        s_pred = ner_spans(y_pred[i])

        recall_sum += recall(s_pred, s)
        precision_sum += precision(s_pred, s)
        
        iterations += 1

        if first:
          debug_print(labels)
          debug_print(s)
          debug_print(y_pred[i])
          debug_print(s_pred)
          debug_print(f"Recall: {recall_sum}")
          debug_print(f"Precision: {precision_sum}")
          first = False

  eval_loss, eval_precision, eval_recall = total_loss / num_loss, _division(precision_sum, iterations), _division(recall_sum, iterations)
  wandb.log({f"{dataset}_loss": eval_loss, f"{dataset}_precision": eval_precision, f"{dataset}_recall": eval_recall, f"{dataset}_epoch": epoch})

  return eval_loss, eval_precision, eval_recall

def _division(n, d):
  return n / d if d else 0

def is_padding(y):
  return y == 9

def is_zero(y):
  return y == 0

def is_entity(y):
  return not is_padding(y) and not is_zero(y)

def ner_spans(tensors : List[torch.tensor]) -> Set[Tuple[int, int, int]]:
  res = set()
  start = -1
  paddingstart = -1
  paddinglength = 0
  label = 0
  tensors = list(map(lambda x: int(x), tensors))

  for i, y in enumerate(tensors):
    if start != -1 and (is_zero(y) or (not is_padding(y) and y % 2 == 1) or i == len(tensors) - 1):
      end = i - paddinglength - 1
      res.add((start, end, label))
      start = -1

    if is_padding(y):
      if paddingstart == -1:
        paddingstart = i
      paddinglength += 1
      continue

    if y % 2 == 1:
      start = paddingstart if paddingstart != -1 else i
      label = y
      if i == len(tensors) - 1:
        res.add((start, i, label))
    paddingstart = -1
    paddinglength = 0

  return res

# def recall(y_pred, y):
#   y_pred = set(map(lambda x: (x[0], x[1]), y_pred))
#   y = set(map(lambda x: (x[0], x[1]), y))
#   recalled = len(y.intersection(y_pred))
#   return recalled / len(y)

def recall(s_pred, s):
  tp = len(s.intersection(s_pred))
  return _division(tp, len(s))

def precision(s_pred, s):
  tp = len(s.intersection(s_pred))
  return _division(tp, len(s_pred))

"""Example:
y:
tensor([9, 0, 9, 3, 0, 0, 0, 0, 0, 0, 9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9,
      0, 0, 0, 3, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:1')
ner_spans: {(10, 11, 3), (2, 3, 3), (27, 27, 3)}

y_pred:
tensor([9, 0, 9, 3, 0, 0, 0, 0, 0, 0, 9, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 9,
        0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:1')
ner_spans: {(2, 3, 3)}

Recall: 0.3333333333333333
Precision: 1.0
"""