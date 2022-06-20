import os
import sys
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from model.resnet import ResNet



if __name__=='__main__':
    print("start training")