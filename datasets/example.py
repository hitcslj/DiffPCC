import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt 
import sys
sys.path.append('/home/guoqing/DiffPC') 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation   
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
# # 创建一个三维numpy数组
# arr = np.array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
#                 [[1, 0, 2], [0, 2, 3], [2, 3, 1]],
#                 [[0, 1, 2], [1, 2, 0], [2, 0, 1]]])

# # 获取值为1的元素的下标
# indices = np.where(arr == 1)
# indices = np.stack(indices, axis=-1)
# print(indices) 

# # 生成三维体素噪声
# noise = np.random.normal(size=(64, 64, 64)) 
# # 创建一个新的图形窗口
# fig = plt.figure() 
# # 创建一个三维轴
# ax = fig.add_subplot(111, projection='3d') 
# # 绘制体素图
# ax.voxels(noise > 0, edgecolor='lightblue')
# ax.axis('off') 
# # 显示图形
# plt.show()
# plt.savefig('noise.png')

# # 创建一个布尔值张量
# bool_tensor = torch.tensor([True, False, True, False]) 
# # 转换为浮点数张量
# float_tensor = bool_tensor.float()
 
# print(float_tensor)
# xx, yy, zz = torch.meshgrid(torch.arange(8), torch.arange(8), torch.arange(8))
# coordinates = torch.stack((xx, yy, zz), -1).reshape(-1, 3)
# print(coordinates) 

def add_noise(point_cloud, noise_level=0.01):
    """
    Adds random noise to a point cloud.

    Parameters:
    point_cloud (numpy.ndarray): The point cloud to add noise to.
    noise_level (float): The standard deviation of the Gaussian noise to add.

    Returns:
    numpy.ndarray: The point cloud with added noise.
    """
    noise = np.random.normal(scale=noise_level, size=point_cloud.shape)
    return point_cloud + noise
def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
pcd_path = '/home/guoqing/DiffPC/data/shapenet_pc'
src_path = '/home/guoqing/DiffPC/data/shapenet_voxel'
npoints = 8192 
nodes = []
import numpy as np
from scipy.ndimage import gaussian_filter
voxel = np.arange(512).reshape(8, 8, 8)  
k = 10
# 将数组展平
flattened = voxel.flatten() 
# 找到前K大值的索引
indices = np.argpartition(flattened, -k)[-k:] 

# 将展平后的索引转换回原始数组的索引
indices = np.unravel_index(indices, voxel.shape)
indices = np.stack(indices, axis=-1) 
nodes = []


pcd = np.loadtxt('/home/guoqing/DiffPC/experiments/OccDiff/ShapeNet55_models/try_to_train_occdiff_airplane_pretrain/pcd_result/0_139_gt.xyz')
print(pcd.shape)
pcd = pc_norm(pcd)
# 找到点云的最小和最大坐标
min_coords = np.min(pcd, axis=0)
max_coords = np.max(pcd, axis=0)

# 计算点云的范围
range_coords = max_coords - min_coords

# 将点云归一化到[0,1]的范围
pcd = (pcd - min_coords) / (range_coords + 1e-3)  

size = 12 
voxel_size = 1.0 / size 
sparse_idx = (pcd // voxel_size).astype(int)   
sparse_voxel = np.zeros((size, size, size))  
np.add.at(sparse_voxel, tuple(sparse_idx.T), 1)
nodes.append(sparse_voxel.flatten()) 
sparse_voxel = np.clip(sparse_voxel, 0, 80) 
sparse_voxel = sparse_voxel / 10
sparse_voxel = np.ceil(sparse_voxel)  
print(np.sum(sparse_voxel > 0))

size = 12 
voxel_size = 1.0 / size 
sparse_idx = np.round(pcd / voxel_size).astype(int)   
sparse_idx = np.clip(sparse_idx, 0, size - 1) 
sparse_voxel2 = np.zeros((size, size, size))  
np.add.at(sparse_voxel2, tuple(sparse_idx.T), 1)
nodes.append(sparse_voxel.flatten()) 
sparse_voxel2 = np.clip(sparse_voxel2, 0, 80) 
sparse_voxel2 = sparse_voxel2 / 10
sparse_voxel2 = np.ceil(sparse_voxel2)  
print(np.sum(sparse_voxel2 > 0))

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(pcd)
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=1.0/12)
    
# 获取体素坐标
sparse_idx = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()]) 
sparse_voxel3 = np.zeros((size+1, size+1, size+1))
np.add.at(sparse_voxel3, tuple(sparse_idx.T), 1)


fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(131, projection='3d')
ax.voxels(sparse_voxel.transpose(0, 2, 1) > 0, edgecolor='k')
ax.axis('off')
ax.grid('off') 
ax = fig.add_subplot(132, projection='3d')
ax.voxels(sparse_voxel2.transpose(0, 2, 1) > 0, edgecolor='k')
ax.axis('off')
ax.grid('off')
ax = fig.add_subplot(133, projection='3d')
ax.voxels(sparse_voxel3.transpose(0, 2, 1) > 0, edgecolor='k')
ax.axis('off')
ax.grid('off')
plt.show()
plt.savefig('./vis/voxel.png')

# with open('/home/guoqing/DiffPC/data/data_split/ShapeNet55-34/ShapeNet-55/airplane/test.txt', 'r') as f:
#     # 读取文件内容
#     content = f.readlines()  
#     for line in content: 
#         pcd = np.load(os.path.join(pcd_path, line[:-1]))
#         pcd = pc_norm(pcd)
#         pcd = add_noise(pcd, 0.02)
#         voxel_data = np.load(os.path.join(src_path, line[:-5] + '.npz'))  
#         dense_idx = voxel_data['dense_voxel']
#         nodes.append(dense_idx.shape[0])
        # fig = plt.figure(figsize=(4, 4))  
        # ax = fig.add_subplot(111, projection='3d')
        # x, z, y = pcd.transpose(1, 0)
        # ax.setxlim = (0, 1)
        # ax.setylim = (0, 1)
        # ax.setzlim = (0, 1)
        # ax.scatter(x, y, z, s=0.5)
        # ax.axis('off') 
        # plt.show()
        # if not os.path.exists('/home/guoqing/DiffPC/datasets/example_img'):
        #     os.makedirs('/home/guoqing/DiffPC/datasets/example_img') 
        # plt.savefig('/home/guoqing/DiffPC/datasets/example_img/' + line[:-4] + '.png')
        # plt.close() 
# fig = plt.figure()
# plt.hist(nodes, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
# plt.xlabel('Number of Points')
# plt.ylabel('Frequency')
# plt.title('Histogram of Points') 
# plt.show()
# plt.savefig('/home/guoqing/DiffPC/datasets/example_img/histogram.png')
# for file in os.listdir(src_path): 
#     if '02691156' not in file:
#         continue 
#     pcd = np.load(os.path.join(pcd_path, file[:-4] + '.npy'))
#     pcd = pc_norm(pcd)   
#     pcd_torch = torch.from_numpy(pcd).cuda().float()
#     pcd_torch = pcd_torch.unsqueeze(0)
#     partial, _ = misc.seprate_point_cloud(pcd_torch, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
#     min_coords = torch.min(pcd_torch, dim=1, keepdim=True)[0] # B, 1, 3
#     max_coords = torch.max(pcd_torch, dim=1, keepdim=True)[0] # B, 1, 3
#     range_coords = max_coords - min_coords
#     pcd_torch = (pcd_torch - min_coords) / (range_coords + 1e-3)
#     partial = (partial - min_coords) / (range_coords + 1e-3) 
#     partial = partial.squeeze(0)
    
    # size = 12  
    # voxel_size = 1.0 / size 
    # partial_idx = torch.div(partial, voxel_size, rounding_mode='floor').long()
    # partial_idx = torch.unique(partial_idx, dim=0)  
    # partial_voxel = torch.zeros((size, size, size))  
    # partial_voxel[partial_idx[:,0], partial_idx[:,1], partial_idx[:,2]] = 1  
    # partial_voxel = partial_voxel.cpu().numpy()
    
    # voxel_data = np.load(os.path.join(src_path, file)) 
    # sparse_voxel = voxel_data['sparse_voxel']   
    # sparse_voxel = adaptive_threshold_3d_gaussian(sparse_voxel, 1.0)
    # dense_idx = voxel_data['dense_voxel']
    # dense_voxel = np.zeros((48, 48, 48)) 
    # dense_voxel[dense_idx[:,0], dense_idx[:,1], dense_idx[:,2]] = 1 
    # dense_pc = dense_idx * (1.0 / 48) + (1.0 / 48) / 2
    # dense_pc = torch.from_numpy(dense_pc).cuda().float()
    # dense_pc = dense_pc.unsqueeze(0)
    # print(np.sum(sparse_voxel > 0), dense_pc.shape, 'L1:', ChamferDistanceL1()(pcd_torch, dense_pc) * 1000, 'L2:', ChamferDistanceL2()(pcd_torch, dense_pc) * 1000)
    
    # nodes.append(np.sum(sparse_voxel > 0))
    # norm = plt.Normalize(sparse_voxel.min(), sparse_voxel.max()) 
    # fig = plt.figure(figsize=(8, 4))  
    # # 使用voxels方法来可视化体素 
    # ax = fig.add_subplot(121, projection='3d')  
    # colors = plt.cm.viridis(norm(sparse_voxel)) 
    # # The stride in the following command can be adjusted to create different visualizations
    # sparse_voxel = sparse_voxel > 0
    # ax.voxels(sparse_voxel0.transpose(0, 2, 1) , facecolors=colors, edgecolor='k', linewidth=0.5)
    # ax.axis('off')   
  
    # ax = fig.add_subplot(132, projection='3d')
    # dense_pc = dense_pc.squeeze(0).cpu().numpy()
    # x, z, y = dense_pc.transpose(1, 0)
    # ax.setxlim = (0, 1)
    # ax.setylim = (0, 1)
    # ax.setzlim = (0, 1)
    # ax.scatter(x, y, z, s=0.5)
    # ax.axis('off') 
    
    # ax = fig.add_subplot(133, projection='3d')
    # x, z, y = pcd.transpose(1, 0)
    # ax.setxlim = (0, 1)
    # ax.setylim = (0, 1)
    # ax.setzlim = (0, 1)
    # ax.scatter(x, y, z, s=0.5)
    # ax.axis('off')
    
    # plt.show()
    # if not os.path.exists('/home/guoqing/DiffPC/datasets/example_img'):
    #     os.makedirs('/home/guoqing/DiffPC/datasets/example_img') 
    # plt.savefig('/home/guoqing/DiffPC/datasets/example_img/' + file[:-4] + '.png')
    # plt.close() 

# fig = plt.figure()
# plt.hist(nodes, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
# plt.xlabel('Number of Occupancy')
# plt.ylabel('Frequency')
# plt.title('Histogram of Points')
# plt.show()
# plt.savefig('/home/guoqing/DiffPC/datasets/example_img/histogram.png')
# plt.close()
# voxel = np.zeros((8, 8, 8))
# n = 0
# for i in range(8):
#     for j in range(8):
#         for k in range(8):
#                 voxel[i, j, k] = n
#                 n += 1
# x = torch.arange(8)
# y = torch.arange(8)
# z = torch.arange(8)
# xx, yy, zz = torch.meshgrid(x, y, z)
# coordinates = torch.stack((xx, yy, zz), -1)
# print(coordinates)
# voxel = torch.arange(512).reshape(8, 8, 8)
# for i in range(512):
#     print(i, torch.where(voxel == i))