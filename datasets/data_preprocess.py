import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt  
def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc  

nodes = []
# src_path = '/home/guoqing/data/ShapeNetCore.v1/ShapeNetCore.v1'
src_path = '/home/guoqing/DiffPC/data/shapenet_pc'
tar_path = '/home/guoqing/DiffPC/data/shapenet_voxel'
for file in os.listdir(src_path): 
        if '02691156' not in file:
            continue
        print(file)
        pcd = np.load(os.path.join(src_path, file))
        pcd = pc_norm(pcd)   
        # 找到点云的最小和最大坐标
        min_coords = np.min(pcd, axis=0)
        max_coords = np.max(pcd, axis=0)

        # 计算点云的范围
        range_coords = max_coords - min_coords

        # 将点云归一化到[0,1]的范围
        pcd = (pcd - min_coords) / (range_coords + 1e-3)
        
        print(np.min(pcd, axis=0), np.max(pcd, axis=0))
         
        size = 72       
        voxel_size = 1.0 / size   
        dense_idx = np.round(pcd / voxel_size).astype(int)
        dense_idx = np.clip(dense_idx, 0, size - 1)
        dense_voxel = np.zeros((size, size, size))
        dense_idx = np.unique(dense_idx, axis=0)  
        dense_voxel[dense_idx[:,0], dense_idx[:,1], dense_idx[:,2]] = 1 
        
        # single occupancy
        size = 24 
        voxel_size = 1.0 / size  
        sparse_idx = np.round(pcd / voxel_size).astype(int)
        sparse_idx = np.clip(sparse_idx, 0, size - 1) 
        sparse_idx = np.unique(sparse_idx, axis=0)
        sparse_idx = sparse_idx // 2  
        sparse_voxel = np.zeros((size // 2, size // 2, size // 2)) 
        np.add.at(sparse_voxel, tuple(sparse_idx.T), 1)   
         
        nodes.append(np.sum(dense_voxel > 0)) 
        # nodes.append(np.sum(sparse_voxel > 0)) 
         
        # np.savez(os.path.join(tar_path, file[:-4] + '.npz'), sparse_voxel=sparse_voxel, dense_voxel=dense_idx)      
         
        # 创建一个新的3D图形 
        # if not os.path.exists('/home/guoqing/DiffPC/datasets/example_img/'):
        #     os.makedirs('/home/guoqing/DiffPC/datasets/example_img/')
        # fig = plt.figure(figsize=(10, 10))  
        # # 使用voxels方法来可视化体素 
        # ax = fig.add_subplot(121, projection='3d')  
        # norm = plt.Normalize(sparse_voxel.min(), sparse_voxel.max())  
        # sparse_voxel = sparse_voxel.transpose(0, 2, 1)  
        # colors = plt.cm.viridis(norm(sparse_voxel)) 
        # # The stride in the following command can be adjusted to create different visualizations
        # sparse_voxel = sparse_voxel > 0
        # ax.voxels(sparse_voxel, facecolors=colors, edgecolor='k', linewidth=0.5)
        # ax.axis('off') 
        
        # ax = fig.add_subplot(122, projection='3d')
        # norm = plt.Normalize(sparse_voxel2.min(), sparse_voxel2.max())
        # sparse_voxel2 = sparse_voxel2.transpose(0, 2, 1)
        # colors = plt.cm.viridis(norm(sparse_voxel2))
        # sparse_voxel2 = sparse_voxel2 > 0
        # ax.voxels(sparse_voxel2, facecolors=colors, edgecolor='k', linewidth=0.5)
        # ax.axis('off') 
      
        # plt.savefig('/home/guoqing/DiffPC/datasets/example_img/' + file + '_sparse.png')
        # plt.close()   

# nodes = np.concatenate(nodes, axis=0)
fig = plt.figure(figsize=(10, 10))
plt.hist(nodes, bins=100)
plt.show()
plt.savefig('/home/guoqing/DiffPC/datasets/example_img/hist.png')