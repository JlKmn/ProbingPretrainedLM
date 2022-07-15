import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from src.models.dataset.tagging_dataset import TaggingDataset
from src.globals import device

def collate_fn(items):
  # items = [(tensor([ 101, 7384, ...19,  100]), tensor([ 0,  1,  6, ..., 11,  0])), ...]
  # max word length of sentences
  max_len = max(len(item[0]) for item in items)

  # sentences = tensor([[0, 0, 0,  ..., 0, 0, 0],[...]...]
  sentences = torch.zeros((len(items), max_len), device=items[0][0].device).long().to(device)
  # taggings = tensor([[0, 0, 0,  ..., 0, 0, 0]
  taggings = torch.zeros((len(items), max_len)).long().to(device)

  for i, (sentence, tagging) in enumerate(items):
    # end of sentences contains tensor contains zeros if len < max_len
    sentences[i][0:len(sentence)] = sentence
    taggings[i][0:len(tagging)] = tagging

  return sentences, taggings

def data_loader(sentences_ids, taggings_ids, batch_size):
    ds = TaggingDataset(sentences_ids, taggings_ids)
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)