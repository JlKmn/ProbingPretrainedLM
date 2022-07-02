from datasets import load_dataset

def load(data_type):
  dataset = load_dataset("conll2003")
  return dataset[data_type]["tokens"], dataset[data_type]["ner_tags"]