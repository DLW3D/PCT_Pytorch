from os.path import join
from helper_ply import read_ply
from helper_tool import ConfigS3DIS as cfg
from helper_tool import DataProcessing as DP
import numpy as np
import torch
import time, pickle, argparse, glob, os
from torch.utils.data import Dataset


class S3DIS(Dataset):
    def __init__(self, test_area_idx, num_points, partition='train'):
        self.split = partition
        self.num_points = num_points
        self.name = 'S3DIS'
        self.path = '/data/S3DIS'
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.num_classes = len(self.label_to_names)  # 13
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])  # array([ 0,  1,  2,  3,  4,  ...
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}  # ｛0：0, 1: 1, 2: 2, 3: 3, ...
        self.ignored_labels = np.array([])

        self.val_split = 'Area_' + str(test_area_idx)
        self.all_files = glob.glob(join(self.path, 'original_ply', '*.ply'))  # 获取所有点云路径

        # Initiate containers S：场景数量，P：原始点云数量，N：下采样后点云数量
        self.val_proj = []  # 验证集原始点云投影 Sv*P
        self.val_labels = []  # 验证集原始点云标签 Sv*P
        self.input_trees = []  # KDTree S
        self.input_colors = []  # RGB S*N*3
        self.input_feature = []  # XYZRGB S,N,6
        self.input_labels = []  # 标签 S*N*1
        self.input_names = []  # 区域名_场景名 S
        self.load_sub_sampled_clouds(cfg.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.replace('/', '\\').split('\\')[-1][:-4]  # 获取点云名（去除路径去除后缀）
            if self.val_split in cloud_name:  # 选中的为验证组，其余为训练组
                cloud_split = 'val'
            else:
                cloud_split = 'train'

            if cloud_split != self.split:  # 跳过非选中组
                continue

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))  # 加载KD树
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))           # 加载下采样后的点云数据

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T  # N*3
            sub_labels = data['class']  # N*1

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            # self.input_trees += [search_tree.data.astype(np.float32)]
            # self.input_colors += [sub_colors.astype(np.float32)]
            self.input_feature += [np.concatenate([np.array(search_tree.data, copy=False, dtype=np.float32),
                                                   sub_colors.astype(np.float32)], axis=-1)]
            self.input_labels += [sub_labels.astype(np.int64)]
            self.input_names += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            # print(
            #     '{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.replace('/', '\\').split('\\')[-1], size * 1e-6,
            #                                               time.time() - t0))

        print('Data Loaded Done!\n')  # 为测试准备重投影的指标

    def __getitem__(self, item):
        pointcloud = self.input_feature[item]
        label = self.input_labels[item]
        # 打乱顺序并采样
        idx = np.arange(len(pointcloud))
        np.random.shuffle(idx)
        idx = idx[:self.num_points]
        pointcloud = pointcloud[idx]
        label = label[idx]
        return pointcloud, label

    def __len__(self):
        return len(self.input_feature)


if __name__ == '__main__':  # 测试用
    train = S3DIS(5, 1024)
    test = S3DIS(5, 1024, 'val')
    for data, label in train:
        print(data.shape)
        print(label.shape)

