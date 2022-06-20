import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from model.resnet import ResNet
from model.resnet import BasicBlock
from model.resnet import Bottleneck

from model.loss_helper import get_loss

from radgraph.radgraph import Radgraph

BATCH_SIZE = 32
MAX_EPOCH = 100
BASE_LEARNING_RATE = 0.001
lr_decay_steps = '80,120,160'
lr_decay_rates = '0.5,0.4,0.3'
LR_DECAY_STEPS = [int(x) for x in lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in lr_decay_rates.split(',')]
WEIGHT_DECAY = 0
IS_AUGMENT = False

TRAIN_DATASET = Radgraph(True, IS_AUGMENT)
TEST_DATASET = Radgraph(False, False)
print(len(TRAIN_DATASET), len(TEST_DATASET))

TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, drop_last=True)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, drop_last=True)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = ResNet(BasicBlock, [3, 4, 6, 3])  # resnet34
net.to(device)

criterion = get_loss
optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

start_epoch = 0

writer = SummaryWriter("log/loss")

def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i, lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch():
    train_running_loss = 0.0
    stat_dict = {}  # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()  # set model to training mode
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        optimizer.zero_grad()
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        end_points = net(inputs)

        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, None)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 96
        if (batch_idx + 1) % batch_interval == 0:
            writer.add_scalars('train', {key: stat_dict[key] / batch_interval for key in stat_dict},
                               (EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx) * BATCH_SIZE)
            print(' ---- batch: %03d ----' % (batch_idx + 1))

            for key in sorted(stat_dict.keys()):
                print('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0


def evaluate_one_epoch():
    stat_dict = {}  # collect statistics
    net.eval()  # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        # if batch_idx % 10 == 0:
        # print('Eval batch: %d'%(batch_idx))
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            end_points = net(inputs)

        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, None)
    return 0

def train(start_epoch):
    global EPOCH_CNT
    min_loss = 1e10
    loss = 0
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        print('**** EPOCH %03d ****' % (epoch))
        print('Current learning rate: %f'%(get_current_lr(epoch)))
        print(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_one_epoch()
        if EPOCH_CNT % 10 == 9: # Eval every 10 epochs
            loss = evaluate_one_epoch()

if __name__ == '__main__':
    # train(start_epoch)
    print("start training")
