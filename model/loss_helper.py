import torch
import torch.nn as nn

def get_loss(pre_label, gt_label):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    POS_WEIGHT = torch.ones(124) * 3.0
    POS_WEIGHT = POS_WEIGHT.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)
    loss = criterion(pre_label, gt_label)
    return loss