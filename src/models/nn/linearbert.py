from transformers import AutoModel
import torch
import torch.nn as nn
from src.globals import device

class LinearBert(nn.Module):
  def __init__(self, num_labels):
    super().__init__()
    self.bert = AutoModel.from_pretrained('bert-base-cased')
    self.probe = nn.Linear(self.bert.config.hidden_size, num_labels)
    self.to(device)

  def forward(self, sentences):
    word_rep, sentence_rep = self.bert(sentences, return_dict=False)
    return self.probe(word_rep)