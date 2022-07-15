import re
import torch
from transformers import AutoTokenizer
from src.globals import tokenizer, debug_print, label_vocab, device

def align_to_bert_tokenization(sentences, labels):
    tokenized_sentences = []
    aligned_labels = []

    for s, l in zip(sentences, labels):
        # s = ['Al', '-', 'Zaman', ':', 'American', 'forces', 'killed', ...]
        # l = ['NNP', 'HYPH', 'NNP', ':', 'JJ', 'NNS', 'VBD', 'NNP', ...]
        # tokenized_sentence = ['Al', '-', 'Z', '##aman', ':', 'American', 'forces', 'killed', ...]
        tokenized_sentence = tokenizer.tokenize(' '.join(s)) 
        aligned_label = []
        current_word = ''
        i = 0
        for token in tokenized_sentence:
            # 1. iteration current_word = Z
            # 2. iteration current_word = Zaman
            current_word += re.sub(r'^##', '', token)
            s[i] = s[i].replace('\xad', '')
            
            assert token == '[UNK]' or s[i].startswith(current_word)

            if token == '[UNK]' or s[i] == current_word:
            # 2. iteration Zaman == Zaman
                current_word = ''
                aligned_label.append(l[i])
                i += 1
            else:
            # 1. iteration Z != Zaman
                aligned_label.append('<pad>')
        
        assert len(tokenized_sentence) == len(aligned_label)

        tokenized_sentences.append(tokenized_sentence)
        aligned_labels.append(aligned_label)
    
    return tokenized_sentences, aligned_labels


def convert_to_ids(sentences, taggings):
  sentences_ids = []
  taggings_ids = []
  for sentence, tagging in zip(sentences, taggings):
    # sentence = ['Al', '-', 'Z', '##aman', ':', 'American', 'forces', 'killed', ...]
    # tagging = ['NNP', 'HYPH', '<pad>', 'NNP', ':', 'JJ', 'NNS', 'VBD', '<pad>', ...]

    # sentence_tensor = [  101,  2586,   118,   163, 19853,   131,  1237,  2088, ...]
    sentence_tensor = torch.tensor(tokenizer.convert_tokens_to_ids(['[CLS]'] + sentence + ['SEP'])).long()
    # tagging_tensor = [ 0,  1,  2,  0,  1,  3,  4,  5,  6,  0,  0,  1,  1,  1, ...]
    tagging_tensor = torch.tensor([0] + [label_vocab[tag] for tag in tagging] + [0]).long()

    sentences_ids.append(sentence_tensor.to(device))
    taggings_ids.append(tagging_tensor.to(device))
  return sentences_ids, taggings_ids

def tokenize(sentences, labels):
    index = 44
    debug_print(f"Sentence {sentences[index]}")
    debug_print(f"Labels {labels[index]}")
    bert_tokenized_sentences, aligned_taggings = align_to_bert_tokenization(sentences, labels)
    debug_print(f"Tokenized Sentence {bert_tokenized_sentences[index]}")
    debug_print(f"Aligned Labels {aligned_taggings[index]}")
    sentences_ids, taggings_ids = convert_to_ids(bert_tokenized_sentences, aligned_taggings)
    debug_print(f"Tensor Sentence {sentences_ids[index]}")
    debug_print(f"Tagging Ids {taggings_ids[index]}")
    debug_print(f'num labels: {len(label_vocab)}')
    return sentences_ids, taggings_ids