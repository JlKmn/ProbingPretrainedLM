from transformers import AutoModel

import torch
import torch.nn as nn
from globals import device

class MultilayerProbeBert(nn.Module):
  def __init__(self, num_labels):
    super().__init__()
    self.bert = AutoModel.from_pretrained('bert-base-cased')
    self.layers = nn.Sequential(
      nn.Linear(self.bert.config.hidden_size, 64),
      nn.ReLU(),
      nn.Linear(64, self.bert.config.hidden_size),
      nn.ReLU(),
      nn.Linear(self.bert.config.hidden_size, num_labels)
    )
    self.to(device)
  
  def forward(self, sentences):
    # We do not want to train the underlying Bert model but only the Linear layer on top
    # Do not back propagate gradients to bert
    with torch.no_grad():
      word_rep, sentence_rep = self.bert(sentences, return_dict=False)
    return self.layers(word_rep)