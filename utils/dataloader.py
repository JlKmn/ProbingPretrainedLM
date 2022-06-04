from torch.utils.data import DataLoader
from models.dataset.postagging import PosTaggingDataset
from globals import *

def collate_fn(items):
  max_len = max(len(item[0]) for item in items)

  sentences = torch.zeros((len(items), max_len), device=items[0][0].device).long().to(device)
  taggings = torch.zeros((len(items), max_len)).long().to(device)

  for i, (sentence, tagging) in enumerate(items):
    sentences[i][0:len(sentence)] = sentence
    taggings[i][0:len(tagging)] = tagging

  return sentences, taggings

def data_loader(sentences_ids, taggings_ids):
    return DataLoader(PosTaggingDataset(sentences_ids, taggings_ids), batch_size=batch_size, collate_fn=collate_fn, shuffle=True)