import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from model.resnet import BasicBlock
from model.resnet import Bottleneck
from model.detr import build

from datasets.radgraph.radgraph import Radgraph

from util.eval_helper import compute_precision_recall, compute_average_precision

import argparse

parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_backbone', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--lr_drop', default=200, type=int)
parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

# * Backbone
parser.add_argument('--backbone', default='resnet18', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--block', default=BasicBlock, help="Resnet block")
parser.add_argument('--channel_list', default=[2, 2, 2, 2], help="Resnet number of channels")

# * Transformer
parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=2048, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=30, type=int,
                    help="Number of query slots")
parser.add_argument('--pre_norm', action='store_true')

# * Matcher
parser.add_argument('--set_cost_disease', default=5, type=float,
                    help="disease coefficient in the matching cost")
parser.add_argument('--set_cost_organ', default=2, type=float,
                    help="organ coefficient in the matching cost")
parser.add_argument('--set_cost_location', default=1, type=float,
                    help="location coefficient in the matching cost")
# * Loss coefficients
parser.add_argument('--disease_loss_coef', default=1, type=float)
parser.add_argument('--organ_loss_coef', default=1, type=float)
parser.add_argument('--location_loss_coef', default=1, type=float)
parser.add_argument('--eos_coef', default=0.001, type=float,
                    help="Relative classification weight of the no-object class")

# dataset parameters
parser.add_argument('--dataset_file', default='radgraph')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--batch_size', default=8)

args = parser.parse_args()

IS_AUGMENT = False

TRAIN_DATASET = Radgraph(True, IS_AUGMENT)
TEST_DATASET = Radgraph(False, IS_AUGMENT)
print(len(TRAIN_DATASET), len(TEST_DATASET))

TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, drop_last=False)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, criterion = build(args)
model.to(device)

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


def train_one_epoch():
    model.train()
    criterion.train()
    train_running_loss = 0.0
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        # Forward pass
        optimizer.zero_grad()
        inputs = batch_data_label['imgs_ls'].to(device)
        targets = []
        for i in range(inputs.shape[0]):
            target = {}
            num_dis = batch_data_label['num_disease'][i]  # list [4,5]
            for k, v in batch_data_label.items():
                value = v[i]
                if k != "imgs_ls" and k != "num_disease":
                    target[k] = value[:num_dis.item()].to(device)
            targets.append(target)
        outputs = model(inputs)
        loss_dict, _, _ = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        train_running_loss += losses.item()
        losses.backward()
        # if args.clip_max_norm > 0:
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()
    return train_running_loss / len(TRAIN_DATALOADER)


@torch.no_grad()
def evaluate():
    # model.eval()
    # criterion.eval()
    val_stat_dict = {}
    val_running_loss = 0.0
    AP = np.zeros(126)  # 126 = num_diseases
    AP_organ = np.zeros(126)
    AP_organ_location = np.zeros(126)
    denominator = np.zeros(126) + 1e-5  # to calculate average among all batches
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        inputs = batch_data_label['imgs_ls'].to(device)
        targets = []
        for i in range(inputs.shape[0]):
            target = {}
            num_dis = batch_data_label['num_disease'][i]  # list [4,5]
            for k, v in batch_data_label.items():
                value = v[i]
                if k != "imgs_ls" and k != "num_disease":
                    target[k] = value[:num_dis.item()].to(device)
            targets.append(target)
        outputs = model(inputs)
        loss_dict, indices, idx = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        val_running_loss += losses.item()

        # eval metrics
        target_diseases_o = torch.cat([t["diseases"][J] for t, (_, J) in zip(targets, indices)])
        target_diseases = torch.full(outputs['pred_disease_logits'].shape[:2], -1,
                                     dtype=torch.int64, device=outputs['pred_disease_logits'].device)  # [B,num_queries]
        target_diseases[idx] = target_diseases_o
        target_organs_o = torch.cat([t["organs"][J] for t, (_, J) in zip(targets, indices)])
        target_organs = torch.full(outputs['pred_organ_logits'].shape[:2], -1,
                                   dtype=torch.int64, device=outputs['pred_organ_logits'].device)
        target_organs[idx] = target_organs_o

        target_locations_o = torch.cat([t["locations"][J] for t, (_, J) in zip(targets, indices)])
        target_locations = torch.full(outputs['pred_location_logits'].shape[:2], -1,
                                      dtype=torch.int64, device=outputs['pred_location_logits'].device)
        target_locations[idx] = target_locations_o

        pred_diseases = torch.argmax(outputs['pred_disease_logits'].softmax(-1), dim=-1)  # b, num_queries, num_diseases
        pred_organs = torch.argmax(outputs['pred_organ_logits'].softmax(-1), dim=-1)
        pred_locations = torch.argmax(outputs['pred_location_logits'].softmax(-1), dim=-1)
        for i in range(126):  # iterate each disease
            for j in range(pred_diseases.shape[0]):  # for each batch
                confidence = []
                index = []
                organ = []
                location = []
                for k in range(pred_diseases.shape[1]):  # each query
                    if pred_diseases[j, k] == i:
                        confidence.append(outputs['pred_disease_logits'].softmax(-1)[j, k, i].detach().cpu())
                        index.append(k)
                        organ.append(pred_organs[j, k].detach().cpu())
                        location.append(pred_locations[j, k].detach().cpu())
                if len(confidence) != 0:  # disease is predicted in this batch
                    denominator[i] += 1
                    confidence = torch.FloatTensor(confidence)
                    sorted_confidence, sorted_index = torch.sort(confidence, descending=True)
                    index = np.array(index, dtype=np.int32)[sorted_index]
                    if len(confidence) == 1:  # to make sure it is still array/list when index only contains 1 element
                        index_list = [index]
                        index = index_list

                    tp = []  # true when disease correct
                    fp = []
                    tp_organ = []  # true when disease and organ correct
                    fp_organ = []
                    tp_organ_location = []  # true when disease, organ and location correct
                    fp_organ_location = []
                    for l in range(len(confidence)):
                        if pred_diseases[j, index[l]] == target_diseases[j, index[l]]:
                            tp.append(1.0)
                            fp.append(0.0)
                        else:
                            tp.append(0.0)
                            fp.append(1.0)

                        if pred_diseases[j, index[l]] == target_diseases[j, index[l]] and \
                                pred_organs[j, index[l]] == target_organs[j, index[l]]:
                            tp_organ.append(1.0)
                            fp_organ.append(0.0)
                        else:
                            tp_organ.append(0.0)
                            fp_organ.append(1.0)

                        if pred_diseases[j, index[l]] == target_diseases[j, index[l]] and \
                                pred_organs[j, index[l]] == target_organs[j, index[l]] and \
                                pred_locations[j, index[l]] == target_locations[j, index[l]]:
                            tp_organ_location.append(1.0)
                            fp_organ_location.append(0.0)
                        else:
                            tp_organ_location.append(0.0)
                            fp_organ_location.append(1.0)

                    precision, recall = compute_precision_recall(np.array(tp), np.array(fp), len(tp))
                    precision_organ, recall_organ = compute_precision_recall(np.array(tp_organ), np.array(fp_organ),
                                                                             len(tp_organ))
                    precision_organ_location, recall_organ_location = compute_precision_recall(
                        np.array(tp_organ_location), np.array(fp_organ_location), len(tp_organ_location))
                    AP[i] += compute_average_precision(precision, recall)
                    AP_organ[i] += compute_average_precision(precision_organ, recall_organ)
                    AP_organ_location[i] += compute_average_precision(precision_organ_location, recall_organ_location)
                else:
                    continue

        print("AP only considering disease:")
        print(np.array(AP) / denominator)
        print("AP considering both disease and organ")
        print(np.array(AP_organ) / denominator)
        print("AP considering disease, organ and location")
        print(np.array(AP_organ_location) / denominator)

        val_stat_dict['val_loss'] = val_running_loss / len(TEST_DATALOADER)
        val_stat_dict['mAP_disease'] = np.sum(np.array(AP) / denominator) / 124  # two disease are not existing
        val_stat_dict['mAP_disease_organ'] = np.sum(np.array(AP_organ) / denominator) / 124
        val_stat_dict['mAP_disease_organ_location'] = np.sum(np.array(AP_organ_location) / denominator) / 124
        # fn_local = target_diseases_o.shape[0]
        # for j in range(target_diseases.shape[0]):
        #    for k in range(target_diseases.shape[1]):
        #        if target_diseases[j, k] != -1:
        #            if target_diseases[j, k] == pred_diseases[j, k] and \
        #                    target_organs[j, k] == pred_organs[j, k] and \
        #                    target_locations[j, k] == pred_locations[j, k]:
        #                tp += 1
        #                fn_local -= 1
        #            elif target_diseases[j, k] != pred_diseases[j, k] or \
        #                    target_organs[j, k] != pred_organs[j, k] or \
        #                    target_locations[j, k] != pred_locations[j, k]:
        #                fp += 1
        # fn += fn_local

    # val_stat_dict['val_loss'] = val_running_loss / len(TEST_DATALOADER)
    # val_stat_dict['precision'] = tp / (tp + fp + 1e-5)
    # val_stat_dict['recall'] = tp / (tp + fn + 1e-5)
    # if (val_stat_dict['precision'] + val_stat_dict['recall']) == 0:
    #    val_stat_dict['f1-score'] = 0.0
    # else:
    #    val_stat_dict['f1-score'] = 2 * val_stat_dict['precision'] * val_stat_dict['recall'] / (
    #            val_stat_dict['precision'] + val_stat_dict['recall'])

    return val_stat_dict


writer = SummaryWriter("log")


def train():
    global EPOCH_CNT
    for epoch in range(0, args.epochs):
        EPOCH_CNT = epoch
        train_loss = train_one_epoch()
        # lr_scheduler.step()
        test_stats = evaluate()
        print(train_loss)
        print(test_stats)
        writer.add_scalar('train/loss', train_loss, EPOCH_CNT)
        writer.add_scalars('val', {key: test_stats[key] for key in test_stats},
                           EPOCH_CNT)


if __name__ == '__main__':
    print("start training")
    train()
