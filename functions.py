import numpy as np
import torch

# loss function

def cross_entropy_loss_RCF(prediction, label):
    #print (label,label.max(),label.min())
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask==1).float()).float()
    num_negative = torch.sum((mask==0).float()).float()
    #print (num_positive, num_negative)
    mask[mask == 1] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(), weight=mask, reduce=False)
    return torch.sum(cost)
