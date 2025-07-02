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


import math
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.general_utils import strip_symmetric, build_scaling_rotation
import joblib
import ipdb

from arguments import colors

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

import numpy as np


class GaussianModel:
    @torch.no_grad()
    def gs2voxel(self, world2local, min_bound, max_bound, voxel_size, detach_dict):
        point_xyz = detach_dict['_xyz']
        point_xyz = (torch.matmul(world2local[:3, :3], point_xyz.T) + world2local[:3, 3:4]).T
        point_sem = detach_dict['_semantic']
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

    @torch.no_grad()
    def create_from_pcd(self, train_data):
        N = train_data['point_xyz'].shape[0]
        self._xyz = nn.Parameter(train_data['point_xyz'].requires_grad_(True))
        self._rgb = nn.Parameter(train_data['point_rgb'].requires_grad_(True))
        self._opacity = nn.Parameter(self.inverse_opacity_activation(torch.full((N, 1), 0.5, device='cuda')).requires_grad_(True))
        self._semantic = train_data['point_sem'] # N, C

        dist2 = torch.clamp_min(distCUDA2(train_data['point_xyz']), 0.0000001)
        scales = self.scaling_inverse_activation(torch.sqrt(dist2))[...,None].repeat(1, 3)
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        rots = torch.zeros((N, 4), device="cuda")
        rots[:, 0] = 1
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        
        if 'nonground_mask' in train_data.keys():
            self.nonground_mask = train_data['nonground_mask']

        self.world2cam_ls = train_data['world2cam_ls']

    def add_from_pcd(self, train_data, *args, **kwargs):
        self.create_from_pcd(train_data)
        
    @torch.no_grad()
    def fusion(self, K):
        pass            
    
    @property
    def get_xyz(self):
        return self._xyz
        
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
        
    @property
    def get_semantic(self):
        return self._semantic

    @property
    def get_rgb(self):
        return self._rgb

    @property
    def get_features_rest(self):
        return None
    
    @property
    def N(self):
        return self._xyz.shape[0]

    def update_detach_for_async_save_eval(self):
        self.detach_dict = {
            '_xyz': self._xyz.detach(),
            '_rgb': self._rgb.detach(),
            '_opacity': self._opacity.detach(),
            '_scaling': self._scaling.detach(),
            '_rotation': self._rotation.detach(),
            '_semantic': self._semantic,
        }
    
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._rgb.shape[1]):
            l.append('f_dc_{}'.format(i))
        # l += ['f_rest_{}'.format(i) for i in range(45)]
        l.append('opacity')
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        return l
        
    def save_ply(self, point_cloud_dir):
        mkdir_p(point_cloud_dir)

        xyz = self.detach_dict['_xyz'].cpu().numpy()
        normals = np.zeros_like(xyz)
        rgb = self.detach_dict['_rgb'].contiguous().cpu().numpy()
        f_dc = RGB2SH(rgb)
        rotation = self.detach_dict['_rotation'].cpu().numpy()
        opacities = np.ones_like(self.detach_dict['_opacity'].cpu().numpy())
        if self.scaling_inverse_activation == torch.log:
            scale = self.detach_dict['_scaling'].cpu().numpy()
        else:
            scale = torch.log(self.scaling_activation(self.detach_dict['_scaling'])).cpu().numpy()
        N = xyz.shape[0]

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(N, dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(os.path.join(point_cloud_dir, "point_cloud_rgb.ply"))
        
        semantic = torch.argmax(self.detach_dict['_semantic'], dim=1).cpu().numpy()
        self.save_semantic_ply(point_cloud_dir, dtype_full, xyz, normals, opacities, scale, rotation, semantic)
    
    def save_semantic_ply(self, point_cloud_dir, dtype_full, xyz, normals, opacities, scale, rotation, semantic):
        f_dc = self.shs[semantic]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(os.path.join(point_cloud_dir, "point_cloud.ply"))

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            # symm = strip_symmetric(actual_covariance)
            return actual_covariance

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self._xyz = torch.empty(0)
        self._rgb = torch.empty(0)
        self._semantic = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.setup_functions()
        self.static_mask = None
        self.compute_cov3D_python = False
        
        self.colors = colors
        self.shs = RGB2SH(self.colors/255)
    


    def reset_opacity(self):
        pass

    def add_densification_stats(self, viewspace_point_tensor_grad, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor_grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def training_setup(self, training_args):
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init, "name": "xyz"},
            {'params': [self._rgb], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
