import os
import torch


def GetTotalWordCount(sentences):
    total = 0
    for s in sentences:
        total += len(s)
    return total