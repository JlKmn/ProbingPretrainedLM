from transformers import AutoModel
import torch
import torch.nn as nn
from globals import device

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

class ProbeResettedBert(nn.Module):
  def __init__(self, num_labels):
    super().__init__()
    self.bert = AutoModel.from_pretrained('bert-base-cased')
    self.probe = nn.Linear(self.bert.config.hidden_size, num_labels)
    self.bert.apply(weight_reset)
    self.to(device)

  def forward(self, sentences):
    with torch.no_grad():
      word_rep, sentence_rep = self.bert(sentences, return_dict=False)
    return self.probe(word_rep)
