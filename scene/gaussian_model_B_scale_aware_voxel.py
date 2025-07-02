#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene.gaussian_model_A_3DGS import GaussianModel as GaussianModel_Parent
import ipdb
from utils.general_utils import inverse_sigmoid
from torch import nn

class GaussianModel(GaussianModel_Parent):
    def get_covariance(self, scaling, rotation):
        return self.covariance_activation(scaling, 1, rotation)

    def gaussian_spatial_weight_simplified(self, x_i, x_j, sigma): # assume diagonal covariance matrix
        diff = x_i - x_j  
        scaled_diff = diff.pow(2) / (2 * sigma.pow(2).unsqueeze(1))  # (x_i - x_j)^2 / (2 * sigma^2)
        exponent = -scaled_diff.sum(dim=-1)  
        return torch.exp(exponent)

    def gaussian_spatial_weight_full(self, x_i, x_j, scaling, rotation): # full covariance matrix
        Sigma = self.get_covariance(scaling, rotation).unsqueeze(1)  
        
        diff = x_i - x_j  # (N, 3)
        Sigma_inv = torch.inverse(Sigma)  # (N, 1, 3, 3)
        
        # diff^T * Sigma_inv * diff
        # (N, 1, 3) @ (N, 3, 3) -> (N, 1, 3)
        # (N, 1, 3) @ (N, 3, 1) -> (N, 1, 1) -> (N,)
        exponent = -0.5 * (diff.unsqueeze(-2) @ Sigma_inv @ diff.unsqueeze(-1)).squeeze()

        return torch.exp(exponent)

    def gaussian_spatial_weight(self, x_i, x_j, scaling, rotation):
        # return self.gaussian_spatial_weight_full(x_i, x_j, scaling, rotation)
        return self.gaussian_spatial_weight_simplified(x_i, x_j, scaling)

    def local_range_offset_full(self, point_scale, voxel_size):
        point_range = torch.round(point_scale/voxel_size + 1e-3).int() # (N, 3)
        point_range = torch.clamp(point_range, 0, 3)
        if point_range.max() == 0:
            return None, None
        
        x_range = torch.arange(-point_range.max(), point_range.max()+1, device='cuda')
        y_range = torch.arange(-point_range.max(), point_range.max()+1, device='cuda')
        z_range = torch.arange(-point_range.max(), point_range.max()+1, device='cuda')
        offset = torch.stack(torch.meshgrid(x_range, y_range, z_range), dim=3).reshape(-1, 3).float().unsqueeze(0)*voxel_size
        
        return point_range*voxel_size, offset
        
    @torch.no_grad()
    def gs2voxel(self, world2local, min_bound, max_bound, voxel_size, detach_dict):
        point_xyz = detach_dict['_xyz']
        point_xyz = (torch.matmul(world2local[:3, :3], point_xyz.T) + world2local[:3, 3:4]).T
        point_sem = detach_dict['_semantic']
        point_radius = self.scaling_activation(detach_dict['_scaling'])
        point_rotation = self.rotation_activation(detach_dict['_rotation'])
        if min_bound is not None and max_bound is not None:
            valid_mask = (point_xyz >= min_bound).all(1) & (point_xyz < max_bound).all(1)
            point_xyz = point_xyz[valid_mask]
            point_sem = point_sem[valid_mask]
            point_radius = point_radius[valid_mask]
            point_rotation = point_rotation[valid_mask]
        
        point_radius, point_offset = self.local_range_offset_full(point_radius, voxel_size)

        if point_offset is not None:
            xyz_ls = []
            sem_ls = []
            for chunk in range(0, point_xyz.shape[0], 100000):
                point_xyz_chunk = point_xyz[chunk:chunk+100000]
                point_sem_chunk = point_sem[chunk:chunk+100000]
                point_radius_chunk = point_radius[chunk:chunk+100000]
                point_rotation_chunk = point_rotation[chunk:chunk+100000]
                
                point_xyz_base = point_xyz_chunk.unsqueeze(1)
                new_point_xyz = point_offset + point_xyz_base # N, K, 3
                spatial_weight = self.gaussian_spatial_weight(new_point_xyz, point_xyz_base, point_radius_chunk, point_rotation_chunk)
                valid_mask = spatial_weight > 0.6065 # exp(-1/2)=0.6065

                new_point_sem = point_sem_chunk.unsqueeze(1).expand(-1, point_offset.shape[1], -1) # N, K, C
                new_point_sem = new_point_sem * spatial_weight.unsqueeze(2) 

                xyz_ls.append(new_point_xyz[valid_mask])
                sem_ls.append(new_point_sem[valid_mask])
            
            point_xyz = torch.cat(xyz_ls, dim=0)
            point_sem = torch.cat(sem_ls, dim=0)

        if min_bound is not None and max_bound is not None:
            valid_mask = (point_xyz >= min_bound).all(1) & (point_xyz < max_bound).all(1)
            point_xyz = point_xyz[valid_mask]
            point_sem = point_sem[valid_mask]

        voxel_indices = (point_xyz / voxel_size).int()
        unique, unique_inverse = torch.unique(voxel_indices, dim=0, return_inverse=True)
        voxel_indices = unique - (min_bound/voxel_size).int()
        assert torch.all(voxel_indices >= 0)

        voxel_sem = torch.zeros((len(unique), point_sem.shape[1]), dtype=torch.float32, device=point_xyz.device)
        voxel_sem.index_add_(0, unique_inverse, point_sem)
        voxel_cls = torch.argmax(voxel_sem, dim=1)

        torch.cuda.empty_cache()
        return voxel_indices, voxel_cls

    def setup_functions(self):
        super().setup_functions()
        self.scaling_activation = torch.sigmoid
        self.scaling_inverse_activation = inverse_sigmoid

    def create_from_pcd(self, train_data):
        super().create_from_pcd(train_data)
        scales = self.scaling_inverse_activation(torch.full((self.N, 3), 0.3, device='cuda'))
        self._scaling = nn.Parameter(scales.requires_grad_(True))

    