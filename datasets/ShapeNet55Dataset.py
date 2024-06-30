import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
import logging
import json 
    
@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.voxel_path = config.VOXEL_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        
        with open('/home/guoqing/DiffPC/data/shapenet_label_dict.json', 'r') as f:
            self.model_label = json.load(f)

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')
        
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc 

    def Rotation(self, pcd, axis, angle=15): 
        angle = np.random.choice([90, 180, 270]) 
        angle = np.pi * angle / 180
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)  
        if axis == 'x':
            rotation_matrix = np.array([[1, 0, 0], [0, cos_theta, sin_theta], [0, -sin_theta, cos_theta]])
        elif axis == 'y':
            rotation_matrix = np.array([[cos_theta, 0, -sin_theta], [0, 1, 0], [sin_theta, 0, cos_theta]])
        elif axis == 'z':
            rotation_matrix = np.array([[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]])
        else:
            raise ValueError(f'axis should be one of x, y and z, but got {axis}!')
        rotated_pts = pcd @ rotation_matrix 
        return rotated_pts    
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        
        # argument_flag = np.random.rand(1) 
        # if argument_flag > 0.5: 
        #     data = self.Rotation(data, np.random.choice(['x', 'y', 'z'])) 
        # data = self.pc_norm(data)          
       
        # 找到点云的最小和最大坐标
        min_coords = np.min(data, axis=0)
        max_coords = np.max(data, axis=0) 
        # 计算点云的范围
        range_coords = max_coords - min_coords 
        # 将点云归一化到[0,1]的范围
        pcd = (data - min_coords) / (range_coords + 1e-3)  
        
        size = 72       
        dense_idx = np.round(np.clip(pcd * size , 0, size - 1)).astype(np.int32)
        dense_idx = np.unique(dense_idx, axis=0) 
        dense_voxel = np.zeros((size, size, size))
        dense_voxel[dense_idx[:,0], dense_idx[:,1], dense_idx[:,2]] = 1 
         
        
        size = 24   
        sparse_idx = np.round(np.clip(pcd * size , 0, size - 1)).astype(np.int32)
        sparse_idx = np.unique(sparse_idx, axis=0) 
        sparse_idx = sparse_idx // 2   
        sparse_voxel = np.zeros((size // 2, size // 2, size // 2)) 
        np.add.at(sparse_voxel, tuple(sparse_idx.T), 1)
         
        data = torch.from_numpy(data).float() 
        label = np.asarray(self.model_label[sample['taxonomy_id']])
        label = torch.from_numpy(label).float()  
        
        # voxel_data = IO.get(os.path.join(self.voxel_path, sample['file_path'][:-4]+'.npz')) 
        # sparse_voxel = voxel_data['sparse_voxel']   
        # dense_idx = voxel_data['dense_voxel'] 
        # dense_voxel = np.zeros((72, 72, 72))            
        # dense_voxel[dense_idx[:,0], dense_idx[:,1], dense_idx[:,2]] = 1
        sparse_voxel = torch.from_numpy(sparse_voxel).long()
        dense_voxel = torch.from_numpy(dense_voxel).float() 
        dense_voxel = 2 * dense_voxel - 1  # change to -1, 1 
        return sample['taxonomy_id'], sample['model_id'], data, sparse_voxel, dense_voxel, label 
  
    def __len__(self):
        return len(self.file_list)