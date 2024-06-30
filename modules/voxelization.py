import torch
import torch.nn as nn
import sys 
sys.path.append('.')
import modules.functional as F

__all__ = ['Voxelization']


# class Voxelization(nn.Module):
#     def __init__(self, resolution, normalize=True, eps=0):
#         super().__init__()
#         self.r = int(resolution)
#         self.normalize = normalize
#         self.eps = eps

#     def forward(self, features, coords):
#         coords = coords.detach()
#         norm_coords = coords - coords.mean(2, keepdim=True)
#         if self.normalize:
#             norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
#         else:
#             norm_coords = (norm_coords + 1) / 2.0
#         norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
#         vox_coords = torch.round(norm_coords).to(torch.int32)
#         return F.avg_voxelize(features, vox_coords, self.r), norm_coords

#     def extra_repr(self):
#         return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')
    
    
class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=False, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        coords = coords.detach()
        norm_coords = coords
        # norm_coords = coords - coords.mean(2, keepdim=True)
        # if self.normalize:
        #     norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        # else:
        #     norm_coords = (norm_coords + 1) / 2.0
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords).long()
        return F.avg_voxelize(features, vox_coords, self.r), vox_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from torch.autograd import gradcheck

    voxelization = Voxelization(9, normalize=False)
    features = torch.randn(1, 3, 1024).cuda()
    coords = torch.randn(1, 3, 1024).cuda()
    x, voxel_coords = voxelization(features, coords) 
    print(x.shape, voxel_coords.shape)