import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from itertools import islice

from globals import label_vocab, lr, BATCHES_PER_EPOCH

def perf(model, loader):
  criterion = nn.CrossEntropyLoss()
  model.eval() # do not apply training-specific steps such as dropout
  total_loss = correct = num_loss = num_perf = 0
  for x, y in islice(loader, 0, BATCHES_PER_EPOCH):
    with torch.no_grad(): # no need to store computation graph for gradients
      # perform inference and compute loss
      y_scores = model(x)
      loss = criterion(y_scores.view(-1, len(label_vocab)), y.view(-1)) # requires tensors of shape (num-instances, num-labels) and (num-instances)

      # gather loss statistics
      total_loss += loss.item()
      num_loss += 1

      # gather accuracy statistics
      y_pred = torch.max(y_scores, 2)[1] # compute highest-scoring tag
      mask = (y != 0) # ignore <pad> tags
      correct += torch.sum((y_pred == y) * mask) # compute number of correct predictions
      num_perf += torch.sum(mask).item()
  return total_loss / num_loss, correct.item() / num_perf

def fit(model, epochs, train_loader, eval_loader):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  for epoch in range(epochs):
    model.train()
    total_loss = num = 0
    for x, y in islice(train_loader, 0, BATCHES_PER_EPOCH):
    #for x, y in train_loader:
      optimizer.zero_grad() # start accumulating gradients
      y_scores = model(x)
      loss = criterion(y_scores.view(-1, len(label_vocab)), y.view(-1))
      loss.backward() # compute gradients though computation graph
      optimizer.step() # modify model parameters
      total_loss += loss.item()
      num += 1
    print(f"[Epoch {1+epoch}]\nTraining loss {total_loss / num}")
    eval_loss, eval_accuracy = perf(model, eval_loader)
    print(f"Eval loss {eval_loss} Eval accuracy {eval_accuracy}")