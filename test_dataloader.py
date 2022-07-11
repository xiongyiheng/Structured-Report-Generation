from datasets.radgraph.radgraph import Radgraph
from torch.utils.data import DataLoader

TRAIN_DATASET = Radgraph(True, False)
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=2,
                              shuffle=False, num_workers=4, drop_last=True)

for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
    print(batch_data_label['gt_labels'].shape)
    break