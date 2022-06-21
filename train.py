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

BATCH_SIZE = 16
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
                              shuffle=True, num_workers=4, drop_last=True)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, drop_last=False)
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
    # adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()  # set model to training mode
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        optimizer.zero_grad()
        inputs = batch_data_label['imgs_ls']
        gt_label = batch_data_label['gt_labels']
        pre_label = net(inputs)

        loss = criterion(pre_label, gt_label)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
    return train_running_loss / len(TRAIN_DATALOADER)


def evaluate_one_epoch():
    stat_dict = {}  # collect statistics
    net.eval()  # set model to eval mode (for bn and dp)
    val_running_loss = 0.0
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        # if batch_idx % 10 == 0:
        # print('Eval batch: %d'%(batch_idx))
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        inputs = batch_data_label['imgs_ls']
        gt_label = batch_data_label['gt_labels']
        with torch.no_grad():
            pre_label = net(inputs)
            loss = criterion(pre_label, gt_label)
            val_running_loss += loss.item()
            output = torch.sigmoid(pre_label)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0

            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    if gt_label[i, j] == output[i, j] == 1:
                        tp += 1
                    if gt_label[i, j] == 1 and gt_label[i, j] != output[i, j]:
                        fp += 1
                    if gt_label[i, j] == output[i, j] == 0:
                        tn += 1
                    if gt_label[i, j] == 0 and gt_label[i, j] != output[i, j]:
                        fn += 1

    stat_dict['val_loss'] = val_running_loss / len(TEST_DATALOADER)
    stat_dict['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    stat_dict['precision'] = tp / (tp + fp)
    stat_dict['recall'] = tp / (fp + fn)
    stat_dict['f1-score'] = 2 * stat_dict['precision'] * stat_dict['recall'] / (
                stat_dict['precision'] + stat_dict['recall'])

    writer.add_scalars('val', {key: stat_dict[key] for key in stat_dict},
                       EPOCH_CNT)
    print(stat_dict)


def train(start_epoch):
    global EPOCH_CNT
    loss = 0
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        print('**** EPOCH %03d ****' % (epoch))
        print('Current learning rate: %f' % (get_current_lr(epoch)))
        print(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        loss = train_one_epoch()
        writer.add_scalar('train_loss', loss, EPOCH_CNT)
        if EPOCH_CNT % 10 == 9:  # Eval every 10 epochs
            evaluate_one_epoch()


if __name__ == '__main__':
    print("start training")
    train(start_epoch)
