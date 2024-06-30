import torch
import numpy as np
import torch.nn as nn
import spconv.pytorch as spconv 
import torch.nn.functional as F
from timm.models.layers import DropPath,trunc_normal_
from utils.logger import *
from modules.voxelization import Voxelization
from .EdgeConv import DGCNN_Grouper
from .build import MODELS 
    
class VoxelCNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = self.conv_block(in_dim, out_dim)
        
        self.conv2 = self.conv_block(out_dim, out_dim) 
        
        self.conv3 = self.conv_block(out_dim, out_dim)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) 
        x = self.conv3(x)
        return x   
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ) 

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.inc = self.conv_block(in_channels, 64)
        # Encoder
        self.enc1 = self.conv_block(64, 128)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = self.conv_block(128, 256)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = self.conv_block(256, 512) 

        # Decoder 
        self.up2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(512, 256)
        self.up1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(256, 128)

        # Final output
        self.out = nn.Conv3d(128, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = self.inc(x)
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))   

        # Decoder  
        dec2 = self.dec2(torch.cat([self.up2(enc3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))

        return self.out(dec1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )  

class SparseNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SparseNet, self).__init__()
        self.shape = [12, 12, 12]
        self.input_proj = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, 64, 3, indice_key="subm0"),
            nn.BatchNorm1d(64),
            nn.ReLU())
        self.sparse_conv1 = spconv.SparseSequential(
            spconv.SubMConv3d(64, 128, 3, indice_key="subm0"),
            nn.BatchNorm1d(128),
            nn.ReLU())
        self.sparse_conv2 = spconv.SparseSequential(
            spconv.SubMConv3d(128, out_channels, 3, indice_key="subm0"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())  
       
        
    def forward(self, features, indices, batch_size):
        x = spconv.SparseConvTensor(features, indices.int(), self.shape, batch_size)
        x = self.input_proj(x)
        x = self.sparse_conv1(x)
        x = self.sparse_conv2(x) 
        return x.dense()

  
@MODELS.register_module()
class OccNet(nn.Module): 
    def __init__(self, config, **kwargs):
        super().__init__()
        self.embed_dim = config.trans_dim  
      
        self.voxellization = Voxelization(12) 
        self.pointEmbedding = UNet3D(3, self.embed_dim)  
        self.voxelEmbedding = UNet3D(1, self.embed_dim) 
 
        self.voxeldecoder = VoxelCNN(self.embed_dim*2, self.embed_dim*2)   
        
        self.coarse_pred = nn.Sequential(
                nn.Linear(self.embed_dim*2, 256), 
                nn.ReLU(inplace=True),
                nn.Dropout(0.5), 
                nn.Linear(256, 256), 
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 9)
            )

        self.apply(self._init_weights) 

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)   
        
    def forward(self, inpc):
        '''
            inpc : input incomplete point cloud with shape B N(2048) C(3)
        ''' 
        bs, N, C = inpc.size()
        size = 12  
           
        point, point_coor = self.voxellization(inpc.transpose(1, 2), inpc.transpose(1, 2)) # B C size * size * size, B 3 M 
        
        point_coor = point_coor.transpose(1,2).contiguous() # B M 3 
        voxel_coor = point_coor[:, :, 0] * 12 * 12 + point_coor[:, :, 1] * 12 + point_coor[:, :, 2] # B M
        voxel = torch.zeros(bs, 12 * 12 * 12).float().to(inpc.device) # B, size * size * size
        voxel.scatter_add_(1, voxel_coor, torch.ones_like(voxel_coor).float().to(inpc.device)) # B, size * size * size    
        voxel = voxel.reshape(bs, 1, 12, 12, 12) # B 1 size size size    
          
        point_features = self.pointEmbedding(point) # B C size  size  size 
        voxel_features = self.voxelEmbedding(voxel) # B, C, size, size, size   
        
        voxel_features = torch.cat([voxel_features, point_features], dim=1) # B 2C size size size    
        voxel_features = voxel_features.reshape(bs, -1, size, size, size)
        voxel_features = self.voxeldecoder(voxel_features)  
        voxel_features = voxel_features.reshape(bs, voxel_features.shape[1], -1).transpose(1,2) # B, size * size * size, C
        coarse_voxel = self.coarse_pred(voxel_features)   
        coarse_voexl = coarse_voxel.reshape(bs, size, size, size, 9)    
        
        return coarse_voexl
    
 