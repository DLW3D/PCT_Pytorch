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
        self.num_classes = len(self.label_to_names)     # 13
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])    # array([ 0,  1,  2,  3,  4,  ...
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}     # ｛0：0, 1: 1, 2: 2, 3: 3, ...
        self.ignored_labels = np.array([])

        self.val_split = 'Area_' + str(test_area_idx)
        self.all_files = glob.glob(join(self.path, 'original_ply', '*.ply'))    # 获取所有点云路径

        # Initiate containers S：场景数量，P：原始点云数量，N：下采样后点云数量
        self.val_proj = []          # 验证集原始点云投影 Sv*P
        self.val_labels = []        # 验证集原始点云标签 Sv*P
        self.input_trees = []   # KDTree S
        self.input_colors = []  # RGB S*N*3
        self.input_labels = []  # 标签 S*N*1
        self.input_names = []   # 区域名_场景名 S
        self.load_sub_sampled_clouds(cfg.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.replace('/', '\\').split('\\')[-1][:-4]  # 获取点云名（去除路径去除后缀）
            if self.val_split in cloud_name:    # 选中的为验证组，其余为训练组
                cloud_split = 'val'
            else:
                cloud_split = 'train'

            if cloud_split != self.split:    # 跳过非选中组
                continue

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))    # 加载KD树
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))           # 加载下采样后的点云数据

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T    # N*3
            sub_labels = data['class']  # N*1

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees += [search_tree]
            self.input_colors += [sub_colors]
            self.input_labels += [sub_labels]
            self.input_names += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.replace('/', '\\').split('\\')[-1], size * 1e-6, time.time() - t0))

        print('Data Loaded Done!\n')    # 为测试准备重投影的指标

    # Generate the input data flow
    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size

        self.possibility = []
        self.min_possibility = []
        # Random initialize 随机初始化
        for i, tree in enumerate(self.input_colors):
            self.possibility += [np.random.rand(tree.data.shape[0]) * 1e-3]  # 为点云的每个原始点生成一个"概率"
            self.min_possibility += [float(np.min(self.possibility[-1]))]  # 确定最小概率

        def spatially_regular_gen():  # 空间正则生成器
            # Generator loop
            for i in range(num_per_epoch):  # 每epoch训练次数 * batch_size

                # Choose the cloud with the lowest probability 选择存在最低概率点的点云
                cloud_idx = int(np.argmin(self.min_possibility))

                # choose the point with the minimum of possibility in the cloud as query point 选择该点云概率最低的点
                point_ind = np.argmin(self.possibility[cloud_idx])

                # Get all points within the cloud from tree structure 从树结构中获取云中的所有点坐标
                points = np.array(self.input_trees[cloud_idx].data, copy=False)

                # Center point of input region 获取该点云的概率最低点(称为中心点)坐标
                center_point = points[point_ind, :].reshape(1, -1)  # 1*3

                # Add noise to the center point 为中心点坐标添加噪声
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)  # 1*3
                pick_point = center_point + noise.astype(center_point.dtype)

                # Check if the number of points in the selected cloud is less than the predefined num_points
                if len(points) < cfg.num_points:  # 检查点云数量是否足够
                    # Query all points within the cloud 查询云中的所有点
                    queried_idx = self.input_trees[cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    # Query the predefined number of points 查询一定数量的点
                    queried_idx = self.input_trees[cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                # Shuffle index 随机打乱点云顺序
                queried_idx = DP.shuffle_idx(queried_idx)
                # Get corresponding points and colors based on the index 根据索引得到相应的点、颜色、标签
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point  # 点云居中
                queried_pc_colors = self.input_colors[cloud_idx][queried_idx]
                queried_pc_labels = self.input_labels[cloud_idx][queried_idx]

                # Update the possibility of the selected points 更新选择的点的概率，距离中心点越近增加越多
                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[cloud_idx][queried_idx] += delta
                self.min_possibility[cloud_idx] = float(np.min(self.possibility[cloud_idx]))  # 更新概率

                # up_sampled with replacement 若点数不足，进行上采样（随机重复采样）
                if len(points) < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                        DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx,
                                    cfg.num_points)

                return pick_point

        gen_func = spatially_regular_gen
        gen_types = (torch.float32, torch.float32, torch.int32, torch.int32, torch.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None])
        return gen_func, gen_types, gen_shapes
                
    def __getitem__(self, item):
        pointcloud = np.array(self.input_trees[item].data, copy=False)
        label = self.input_labels[item]
        # 打乱顺序
        idx = np.arange(len(pointcloud))
        np.random.shuffle(idx)
        idx = idx[:self.num_points]
        pointcloud = pointcloud[idx]
        label = label[idx]
        return pointcloud, label

    def __len__(self):
        return len(self.input_trees)


if __name__ == '__main__':  # 测试用
    train = S3DIS(1024)
    test = S3DIS(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
