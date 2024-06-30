import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F 
from .build import MODELS

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)
  
class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=16, dim9=False):
        super(EdgeConv, self).__init__()
        self.k = k
        self.dim9 = dim9
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        ) 
 
    def forward(self, x): # (batch_size, in_channels, num_points)
        batch_size = x.size(0)
        num_points = x.size(2)
 
        x = get_graph_feature(x, k=self.k, dim9=self.dim9)  # (batch_size, 2*num_dims, num_points, k)
        x = self.conv(x)                                   # (batch_size, out_channels, num_points, k) 
        x = x.max(dim=-1, keepdim=False)[0]               # (batch_size, out_channels, num_points)
 
        return x
@MODELS.register_module()
class FineNet(nn.Module):
    def __init__(self, config):
        super().__init__() 
        self.local_embedding = EdgeConv(3, 64, k=32)
        self.conv1 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024) 
        self.convs1 = torch.nn.Conv1d(1024+1024+512+128+128+64, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, 3, 1) 
        self.bns1 = nn.BatchNorm1d(256)  
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128) 

    def forward(self, point_cloud):
        B, N, D = point_cloud.shape 
        point_cloud = point_cloud.transpose(2, 1) 
        out0 = self.local_embedding(point_cloud)  
        out1 = F.relu(self.bn1(self.conv1(out0)))    
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))  
        out4 = F.relu(self.bn4(self.conv4(out3)))
        out5 = self.bn5(self.conv5(out4))
        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 1024) 
        expand = out_max.view(-1, 1024, 1).repeat(1, 1, N) 
        concat = torch.cat([expand, out5, out4, out3, out2, out1], 1)  
        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net))) 
        net = F.relu(self.bns3(self.convs3(net)))  
        delta_xyz = self.convs4(net)      
        fine_point_cloud = point_cloud + delta_xyz 
        fine_point_cloud = fine_point_cloud.transpose(2, 1)
        return fine_point_cloud      
 