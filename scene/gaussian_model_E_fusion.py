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
from torch import nn
from scene.gaussian_model_D_track_hist import GaussianModel as GaussianModel_Parent
from custom_utils import query_np_kdtree
import ipdb
import numpy as np
class GaussianModel(GaussianModel_Parent):
    def kl_divergence(self, p, q):
        return torch.sum(p * torch.log(p / q), dim=-1)

    def js_divergence(self, p, q):
        m = 0.5 * (p + q)
        return 0.5 * (self.kl_divergence(p, m) + self.kl_divergence(q, m))

    
    def gaussian_radiometric_weight(self, c_i, c_j, sigma=1):
        def l2(c_i, c_j):
            return (c_i - c_j).pow(2).sum(-1)
        return torch.exp(-l2(c_i, c_j) / (2 * sigma**2))

    def gaussian_semantic_weight(self, p_i, p_j, sigma=1):
        div = self.js_divergence(p_i, p_j)
        gaussian_weight = torch.exp(-div / (2 * sigma**2))
        
        return gaussian_weight

    def inverse_covariance_fusion(self, Sigma1, Sigma2):
        """
        Compute the Inverse Covariance Fusion of two covariance matrices.

        Args:
            Sigma1 (torch.Tensor): The first covariance matrix, shape (N, K, 3, 3).
            Sigma2 (torch.Tensor): The second covariance matrix, shape (N, K, 3, 3).

        Returns:
            torch.Tensor: The fused covariance matrix, shape (N, K, 3, 3).
        """

        Omega1 = torch.inverse(Sigma1)  # (N, K, 3, 3)
        Omega2 = torch.inverse(Sigma2)  # (N, K, 3, 3)

        Sigma_fused_inv = Omega_fused = Omega1 + Omega2  # (N, K, 3, 3)

        # Sigma_fused = torch.inverse(Omega_fused)  # (N, K, 3, 3)

        return Sigma_fused_inv

    def gaussian_spatial_weight_full_fused(self, x_i, x_j, scaling_i, scaling_j, rotation_i, rotation_j):
        """
        Compute the spatial Gaussian weight G_spatial(i, j) using full 3×3 covariance matrices.

        Args:
            x_i (torch.Tensor): Coordinates of point i, shape (N, 3).
            x_j (torch.Tensor): Coordinates of point j, shape (N, 3).
            Sigma (torch.Tensor): Full 3×3 covariance matrix for each point pair, shape (N, 3, 3).

        Returns:
            torch.Tensor: Gaussian weights for each point pair, shape (N,).
        """

        Sigma_i = self.get_covariance(scaling_i, rotation_i).unsqueeze(1)  #  (N, 1, 3, 3)
        n = x_j.shape[0]
        Sigma_j = self.get_covariance(scaling_j.reshape(-1, 3), rotation_j.reshape(-1, 4)).reshape(n, -1, 3, 3)  #  (N, K, 3, 3)
        Sigma_inv = self.inverse_covariance_fusion(Sigma_i, Sigma_j)  #  (N, 3, 3)
        
        diff = x_i - x_j  # (N, 3)
        
        # (N, 1, 3) @ (N, 3, 3) -> (N, 1, 3)
        # (N, 1, 3) @ (N, 3, 1) -> (N, 1, 1) -> (N,)
        exponent = -0.5 * (diff.unsqueeze(-2) @ Sigma_inv @ diff.unsqueeze(-1)).squeeze()

        return torch.exp(exponent)
    
    def gaussian_spatial_weight_simplified_fused(self, x_i, x_j, scaling_i, scaling_j):
        """
        Compute the spatial Gaussian weight G_spatial(i, j) assuming diagonal covariance matrices.

        Args:
            x_i (torch.Tensor): Coordinates of point i, shape (N, 3).
            x_j (torch.Tensor): Coordinates of point j, shape (N, 3).
            scaling_i (torch.Tensor): Spatial scaling of point i (e.g., standard deviations along x/y/z), shape (N, 3).
            scaling_j (torch.Tensor): Spatial scaling of point j, shape (N, 3).

        Returns:
            torch.Tensor: Gaussian weights for each point pair, shape (N,).
        """

        Sigma_inv = 1 / (scaling_i.unsqueeze(1)**2 + scaling_j**2)  # (N, 3)
        diff = x_i - x_j  # (N, 3)

        exponent = -0.5 * (diff**2 * Sigma_inv).sum(dim=-1)  # (N,)

        return torch.exp(exponent)

    def gaussian_spatial_weight_fused(self, x_i, x_j, scaling_i, scaling_j, rotation_i, rotation_j):
        # return self.gaussian_spatial_weight_full_fused(x_i, x_j, scaling_i, scaling_j, rotation_i, rotation_j)
        return self.gaussian_spatial_weight_simplified_fused(x_i, x_j, scaling_i, scaling_j)
    

    @torch.no_grad()
    def fusion(self, K, sigma_c=1, sigma_s=1):
        scaling = self.scaling_activation(self.detach_dict['_scaling'])
        rotation = self.rotation_activation(self.detach_dict['_rotation'])

        xyz_np = self.detach_dict['_xyz'].cpu().numpy()
        xyz_dis, xyz_ind = query_np_kdtree(xyz_np, xyz_np, k=K, max_dist=np.inf) # (N, K)
        if len(xyz_ind.shape) == 1:
            xyz_ind = xyz_ind[:, None]  
        
        xyz_weight = self.gaussian_spatial_weight_fused(
                                                self.detach_dict['_xyz'].unsqueeze(1), # (N, 1, 3) 
                                                self.detach_dict['_xyz'][xyz_ind], # (N, K, 3)
                                                scaling, # (N, 3),
                                                scaling[xyz_ind], # (N, K, 3)
                                                rotation, # (N, 4)
                                                rotation[xyz_ind] # (N, K, 4)
                                            )  # N, K
        
        rgb_weight = self.gaussian_radiometric_weight(self.detach_dict['_rgb'].unsqueeze(1), self.detach_dict['_rgb'][xyz_ind], sigma=sigma_c)  # N, K
        
        neighbor_sem = self.detach_dict['_semantic'][xyz_ind]  # N, K, C
        semantic_weight = self.gaussian_semantic_weight(self.detach_dict['_semantic'].unsqueeze(1), neighbor_sem, sigma=sigma_s)  # N, K
        
        weight = xyz_weight + rgb_weight + semantic_weight  # N, K
        weight /= weight.sum(dim=1, keepdim=True)  # N, K
        
        self.detach_dict['_semantic'] = (neighbor_sem * weight.unsqueeze(-1)).sum(dim=1)  # N, C
        self.detach_dict['_semantic'] /= self.detach_dict['_semantic'].sum(dim=1, keepdim=True)