import numpy as np
import torch
import torch.nn.functional as F

def label_wise_accuracy(output, target):
    pred = (output == output.max(dim=1, keepdim=True).values).int()
    correct = (pred == F.one_hot(target.long(), num_classes=output.shape[1])).float()
    label_accuracy = torch.sum(correct, dim=0)
    return label_accuracy
