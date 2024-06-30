# import torch
# from torch import nn
# import torch.nn.functional as F
# input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
# print('input:', input)
# x = F.interpolate(input, scale_factor=5, mode='nearest')
# print('x:', x)
# import numpy as np

# # 设置体素的分辨率
# resolution = 6

# # 生成索引数组
# x = np.arange(resolution)
# y = np.arange(resolution)
# z = np.arange(resolution)
# xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

# # 将索引数组合并为一个(3, resolution^3)的数组
# indices = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
 

# import torch

# resolution = 6

# # Generate index arrays
# x = torch.arange(resolution)
# y = torch.arange(resolution)
# z = torch.arange(resolution)
# xx, yy, zz = torch.meshgrid(x, y, z)

# # Combine the index arrays into a (resolution^3, 3) array
# indices = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=-1) 


import spconv.pytorch as spconv
from torch import nn
import spconv
import torch
import torch.nn as nn
import torch.nn.functional as F


import spconv.pytorch as spconv
from torch import nn  
 
import torch
if __name__ == '__main__': 
  model = UNet3D(1, 768)
  input = torch.randn(1, 1, 12, 12, 12)
  out1, output = model(input)
  print(out1.shape, output.shape)