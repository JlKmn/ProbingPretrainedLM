import os
import urllib.request
import conllu

from globals import DATASET_DIR, debug_print

def load_conllu(filename):
  with open(filename, encoding="utf-8") as fp:
    data = conllu.parse(fp.read())
  sentences = [[token['form'] for token in sentence] for sentence in data]
  taggings = [[token['xpos'] for token in sentence] for sentence in data]
  return sentences, taggings


def load(filename):
    filePath = f"{DATASET_DIR}/{filename}" 
    if not os.path.exists(filePath):
        urllib.request.urlretrieve('https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/' + filename, filePath)

    sentences, labels = load_conllu(filePath)
    debug_print(list(zip(sentences[0], labels[0])))
    return sentences, labels