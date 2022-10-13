from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, SubsetRandomSampler, ConcatDataset

from sklearn.model_selection import KFold
import numpy as np

from model.resnet import ResNet
from model.resnet import BasicBlock
from model.resnet import Bottleneck

from model.loss_helper import get_loss

from datasets.radgraph.radgraph_baseline import Radgraph

BATCH_SIZE = 16
MAX_EPOCH = 15
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
torch.manual_seed(42)

### k-fold ###
DATASET = ConcatDataset([TRAIN_DATASET, TEST_DATASET])
k = 10
splits = KFold(n_splits=k, shuffle=True, random_state=42)


# net = ResNet(BasicBlock, [2, 2, 2, 2], is_backbone=False)  # resnet34: [3, 4, 6, 3], resnet18: [2, 2, 2, 2]
# net.to(device)
#
# criterion = get_loss
# optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

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


def train_one_epoch(net, device, dataloader, criterion, optimizer):
    train_running_loss = 0.0
    stat_dict = {}  # collect statistics
    # adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()  # set model to training mode
    for batch_idx, batch_data_label in enumerate(dataloader):
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
    return train_running_loss / len(dataloader)


@torch.no_grad()
def evaluate(net, device, dataloader, criterion):
    stat_dict = {}  # collect statistics
    net.eval()  # set model to eval mode (for bn and dp)
    val_running_loss = 0.0
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    for batch_idx, batch_data_label in enumerate(dataloader):
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

    stat_dict['val_loss'] = val_running_loss / len(dataloader)
    stat_dict['accuracy'] = (tp + tn) / (tp + tn + fp + fn+1e-5)
    stat_dict['precision'] = tp / (tp + fp+1e-5)
    stat_dict['recall'] = tp / (tp + fn+1e-5)
    if (stat_dict['precision'] + stat_dict['recall']) == 0:
        stat_dict['f1-score'] = 0.0
    else:
        stat_dict['f1-score'] = 2 * stat_dict['precision'] * stat_dict['recall'] / (
                stat_dict['precision'] + stat_dict['recall'])

    writer.add_scalars('val', {key: stat_dict[key] for key in stat_dict},
                       EPOCH_CNT)

    print(stat_dict)
    return stat_dict



def train(start_epoch):
    global EPOCH_CNT
    prec_ls = np.zeros([1, k])
    accuracy_ls = np.zeros([1, k])
    recall_ls = np.zeros([1, k])
    f1_ls = np.zeros([1, k])
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(DATASET)))):

        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)


        train_loader = DataLoader(DATASET, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4,
                                  drop_last=True)
        test_loader = DataLoader(DATASET, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=4,
                                 drop_last=False)

        net = ResNet(BasicBlock, [2, 2, 2, 2], is_backbone=False)  # resnet34: [3, 4, 6, 3], resnet18: [2, 2, 2, 2]
        net.to(device)

        criterion = get_loss
        optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        loss = 0

        for epoch in range(start_epoch, MAX_EPOCH):
            EPOCH_CNT = epoch
            print('**** EPOCH %03d ****' % (epoch))
            #print('Current learning rate: %f' % (get_current_lr(epoch)))
            #print(str(datetime.now()))
            loss = train_one_epoch(net, device, train_loader, criterion, optimizer)
            writer.add_scalar('train_loss_fold'+str(fold+1), loss, EPOCH_CNT)
            if EPOCH_CNT % 1 == 0:  # Eval every epoch
                stat_dict = evaluate(net, device, test_loader, criterion)
                if stat_dict['accuracy'] > accuracy_ls[0, fold]:
                    accuracy_ls[0, fold] = stat_dict['accuracy']
                if stat_dict['precision'] > prec_ls[0, fold]:
                    prec_ls[0, fold] = stat_dict['precision']
                if stat_dict['recall'] > recall_ls[0, fold]:
                    recall_ls[0, fold] = stat_dict['recall']
                if stat_dict['f1-score'] > f1_ls[0, fold]:
                    f1_ls[0, fold] = stat_dict['f1-score']
        print('fold_performance: accu:{} prec:{} recall:{} f1:{}'.format(accuracy_ls[0, fold],prec_ls[0, fold],recall_ls[0, fold],f1_ls[0, fold]))
    print('avg_preformance: accu:{} prec:{} recall:{} f1:{}'.format(np.mean(accuracy_ls),np.mean(prec_ls),np.mean(recall_ls),np.mean(f1_ls)))

if __name__ == '__main__':
    print("start training")
    train(start_epoch)
