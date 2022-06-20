import os

#import open3d as o3d

import numpy as np

import json

from torch.utils.data import Dataset

class Radgraph(Dataset):
    def __init__(self, is_train, use_augment, img_path="/home/mlmi-matthias/physionet.org/files/mimic-cxr-jpg/2.0.0/files/",label_path = "/home/mlmi-kamilia/jingsong/cnn/radgraph/"):
        # is_train: training set or val set
        # is_augment: augment or not
        # use_height: use height features as the 4th column of pcd. Ref: https://github.com/facebookresearch/votenet/blob/main/scannet/scannet_detection_dataset.py
        super(Radgraph, self).__init__()

        self.is_train = is_train
        self.img_path = img_path
        self.label_path = label_path
        self.is_augment = use_augment

        if self.is_train:
            f = open(self.label_path + "final_dataset_train.json", "r")
        else:
            f = open(self.label_path + "final_dataset_val.json", "r")

        self.index = f.read().splitlines()

        # load the mean size
        MEAN_SIZE = np.load(MEAN_SIZE_PATH)
        self.mean_size = MEAN_SIZE

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        ### return: pcd:x,y,z,height, controid:x,y,z, sizes:x,y,z, idxes: 0-7, mean_size:x,y,z,
        idx_str = self.index[idx]
        scene_path = self.dataset_path + idx_str

        # extract one scan
        scan, choices = get_pcd_from_ply(scene_path, self.sample_size)

        # extract labels & orientation matrix
        centroids, sizes, idxes_rio7, orientation, heading_angles, box_label_mask,_= extract_label(scene_path)

        # get mean_sizes
        mean_sizes = np.zeros((len(idxes_rio7), 3))
        for i in range(len(idxes_rio7)):
            mean_sizes[i]= self.mean_size[idxes_rio7[i]]

        # get mean_size_array
        mean_size_array = self.mean_size + 0

        # data aug
        if self.is_augment:
            #if np.random.random() > 0.5:
                # Flipping along the YZ plane
                #scan[:,0] = -1 * scan[:,0]
                #centroids[:,0] = -1 * centroids[:,0]
                #heading_angles[heading_angles>0.0] = np.pi - heading_angles[heading_angles>0.0]
                #heading_angles[heading_angles<0.0] = -(np.pi + heading_angles[heading_angles<0.0])

            #if np.random.random() > 0.5:
                # Flipping along the XZ plane
                #scan[:,1] = -1 * scan[:,1]
                #centroids[:,1] = -1 * centroids[:,1]
                #heading_angles = -heading_angles

            # Rotation along up-axis/Z-axis
            heading_angles = np.array(heading_angles, dtype=np.float64)
            #theta = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
            theta = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            check_theta = heading_angles + theta
            mask = np.zeros(check_theta.shape)
            mask[check_theta>=np.pi] = 1
            mask[check_theta<-np.pi] = 1
            if np.sum(mask) > 0:
                theta = 0.0
            matrix = np.zeros((3,3))
            c = np.cos(theta)
            s = np.sin(theta)
            matrix[0, 0] = c
            matrix[0, 1] = -s
            matrix[1, 0] = s
            matrix[1, 1] = c
            matrix[2, 2] = 1.0
            scan[:,0:3] = np.dot(scan[:,0:3], np.transpose(matrix))
            centroids[:,0:3] = np.dot(centroids[:,0:3], np.transpose(matrix))
            heading_angles += theta

            # Rescale randomly by 0.9 - 1.1
            proportion = np.random.uniform(0.9, 1.1, 1)
            scan = scan * proportion
            centroids = centroids * proportion
            sizes = sizes * proportion

        # compute vote after data-aug
        vote_label_mask, vote_label = compute_vote(scene_path,choices,scan,centroids)

        # compute the height features
        if self.use_height:
            floor_height = np.percentile(scan[:,2], 0.99)
            height = scan[:,2] - floor_height
            scan = np.concatenate([scan,np.expand_dims(height, 1)], 1)

        # get heading angle bins and heading residuals
        heading_cls = theta_array_to_class(heading_angles)
        heading_residuals = heading_angles - (np.pi / 12 + np.pi / 6 * (heading_cls - 6))

        # get size residuals
        size_residuals = sizes - mean_sizes # 147, 3 - 147, 3

        ret_dict = {}
        ret_dict['point_clouds'] = scan.astype(np.float32)
        ret_dict['center_label'] = centroids.astype(np.float32)
        ret_dict['heading_class_label'] = heading_cls.astype(np.int64)
        ret_dict['heading_residual_label'] = heading_residuals.astype(np.float32)
        ret_dict['size_class_label'] = idxes_rio7.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        target_bboxes_semcls = idxes_rio7
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = box_label_mask.astype(np.float32)
        ret_dict['vote_label'] = vote_label.astype(np.float32)
        ret_dict['vote_label_mask'] = vote_label_mask.astype(np.int64)
        #ret_dict['scan_dict'] = scan_dict
        # ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        # ret_dict['pcl_color'] = pcl_color
        return ret_dict