import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import torch

import json

from torch.utils.data import Dataset

class Radgraph(Dataset):
    def __init__(self, is_train, is_augment, data_path="/home/mlmi-matthias/physionet.org/files/mimic-cxr-jpg/2.0.0/files/",label_path = "/home/mlmi-kamilia/jingsong/StructuredReportGeneration/dataset/"):
        # is_train: training set or val set
        # is_augment: augment or not
        super(Radgraph, self).__init__()

        self.is_train = is_train
        self.data_path = data_path
        self.label_path = label_path
        self.is_augment = is_augment

        if self.is_train:
            with open(self.label_path + "final_dataset_train.json", 'r') as f:
                data = json.load(f)

        else:
            with open(self.label_path + "final_dataset_dev.json", 'r') as f:
                data = json.load(f)

        self.idx = list(data.keys())

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):

        report_id = self.idx[idx]
        #g roup_id: p10, patient_id:p18004941, study_id = s588...
        [group_id,patient_id,study_id] = report_id.split('/')
        [study_id,_] = study_id.split('.')
        self.img_path = self.data_path+group_id+'/'+patient_id+'/'+study_id

        ### extract the jpg files
        img_ls = []
        for file in os.listdir(self.img_path):
            if file.endswith('.jpg'):
                img_ls.append(os.path.join(self.img_path, file))

        ### process the jpg files
        img_ls_tensor = []
        for i in range(len(img_ls)):
            image = Image.open(img_ls[i])
            transform = transforms.Compose([
                transforms.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)).unsqueeze(0)),
                transforms.Resize([2000, 2000]),
                transforms.Normalize([0], [255])
            ])
            a = transform(image)
            img_ls_tensor.append(transform(image))

        ### extract the labels:
        if self.is_train:
            with open(self.label_path + "final_dataset_train.json", 'r') as f:
                data = json.load(f)

        else:
            with open(self.label_path + "final_dataset_dev.json", 'r') as f:
                data = json.load(f)
        lb_ls = data[report_id]["labels"]

        ### output
        rect = {}
        rect['gt_labels'] = torch.FloatTensor(lb_ls)
        #if len(img_ls_tensor) > 1:
        #    from random import randrange
        #    rect['imgs_ls'] = img_ls_tensor[randrange(2)]
        #else:
        #    rect['imgs_ls'] = img_ls_tensor[0]
        rect['imgs_ls'] = img_ls_tensor[0]
        return rect


if __name__ == "__main__":
    Data = Radgraph(is_train=True, is_augment=False, data_path="/home/mlmi-matthias/physionet.org/files/mimic-cxr-jpg/2.0.0/files/",
                    label_path="/home/mlmi-kamilia/jingsong/StructuredReportGeneration/dataset/")