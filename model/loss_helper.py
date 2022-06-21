import torch.nn as nn

POS_WEIGHT = None

def get_loss(pre_label, gt_label):
    criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)
    loss = criterion(pre_label, gt_label)
    return loss