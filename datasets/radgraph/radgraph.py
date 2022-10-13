import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import torch

import json

from torch.utils.data import Dataset

class Radgraph(Dataset):
    def __init__(self, is_train, is_augment, data_path="/home/mlmi-matthias/physionet.org/files/mimic-cxr-jpg/2.0.0/files/",label_path = "/home/mlmi-kamilia/jingsong/StructuredReportGeneration/datasets/radgraph/"):
        # is_train: training set or val set
        # is_augment: augment or not
        super(Radgraph, self).__init__()

        self.is_train = is_train
        self.data_path = data_path
        self.label_path = label_path
        self.is_augment = is_augment
        self.none_obj = [126, 47, 80]  ### the placeholder lst for none_object

        if self.is_train:
            with open(self.label_path + "one_batch.json", 'r') as f:#detr_data_train
                data = json.load(f)

        else:
            with open(self.label_path + "one_batch.json", 'r') as f:#detr_data_dev
                data = json.load(f)

        self.idx = list(data.keys())

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):

        report_id = self.idx[idx]
        #group_id: p10, patient_id:p18004941, study_id = s588...
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
                transforms.Resize([224, 224]),
                transforms.Normalize([0], [255])
            ])
            a = transform(image)
            img_ls_tensor.append(transform(image))

        ### extract the labels:
        if self.is_train:
            with open(self.label_path + "one_batch.json", 'r') as f:#detr_data_train
                data = json.load(f)

        else:
            with open(self.label_path + "one_batch.json", 'r') as f:#detr_data_dev
                data = json.load(f)
        ob_ls = data[report_id]  #list(N,3)
        diseases_ls = []
        organ_ls = []
        locations_ls = []
        num_dis = 0

        #targets = [{'diseases': tensor[1, 5, 3...], 'organs': tensor[6, 3, 7...], 'locations': [4, 77, 90...], 'imgs_ls': IMG_TENSOR}, ......]

        ###
        for ob in ob_ls:
            if ob != self.none_obj:
                num_dis += 1 # count the num of disease
            diseases_ls.append(ob[0])
            organ_ls.append(ob[1])
            locations_ls.append(ob[2])

        ### output
        rect = {}
        rect['diseases'] = torch.LongTensor(diseases_ls)
        rect['organs'] = torch.LongTensor(organ_ls)
        rect['locations'] = torch.LongTensor(locations_ls)
        rect['num_disease'] = num_dis
        #if len(img_ls_tensor) > 1:
        #    from random import randrange
        #    rect['imgs_ls'] = img_ls_tensor[randrange(2)]
        #else:
        #    rect['imgs_ls'] = img_ls_tensor[0]
        rect['imgs_ls'] = img_ls_tensor[0]
        #rect = [rect]
        return rect


if __name__ == "__main__":
    Data = Radgraph(is_train=True, is_augment=False, data_path="/home/mlmi-matthias/physionet.org/files/mimic-cxr-jpg/2.0.0/files/",
                    label_path="/home/mlmi-kamilia/jingsong/StructuredReportGeneration/datasets/")