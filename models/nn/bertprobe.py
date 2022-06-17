from transformers import AutoModel

import torch
import torch.nn as nn

from globals import device

class LinearProbeBert(nn.Module):
  def __init__(self, num_labels):
    super().__init__()
    self.bert = AutoModel.from_pretrained('bert-base-cased')
    # Linear layer that applies y = x * A^T + b transformation
    # Input size is of size of the bert hidden size. Output size is the number of tags
    self.probe = nn.Linear(self.bert.config.hidden_size, num_labels)
    self.to(device)

  def parameters(self):
    return self.probe.parameters()
  
  def forward(self, sentences):
    # We do not want to train the underlying Bert model but only the Linear layer on top
    # Do not back propagate gradients to bert
    with torch.no_grad():
      word_rep, sentence_rep = self.bert(sentences, return_dict=False)
    return self.probe(word_rep)
