import torch
import torch.nn as nn
import torch.nn.functional as F
from src.globals import tokenizer, device

class LinearProbeRandom(nn.Module):
  def __init__(self, num_labels):
    super().__init__()
    self.embedding = nn.Embedding(tokenizer.vocab_size, 768)
    self.probe = nn.Linear(768, num_labels)
    self.to(device)
  
  def forward(self, sentences):
    # Embedding layer needs no back propagation
    with torch.no_grad():
      word_rep = self.embedding(sentences)
    return self.probe(word_rep)