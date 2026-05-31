#
# Unified Gaussian model (formerly A–E ablation chain).
#

import math
import os
import time

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from arguments import colors
from custom_utils import project_to_pixel, query_np_kdtree
from utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except ImportError:
    pass

def icp_flow_open3d(src_pts: torch.Tensor, dst_pts: torch.Tensor, max_correspondence_distance=0.5):
    src_np = src_pts.detach().cpu().numpy()
    dst_np = dst_pts.detach().cpu().numpy()

    pcd_src = o3d.geometry.PointCloud()
    pcd_dst = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src_np)
    pcd_dst.points = o3d.utility.Vector3dVector(dst_np)
    
    centroid_src = np.mean(src_np, axis=0)
    centroid_dst = np.mean(dst_np, axis=0)

    translation = centroid_dst - centroid_src
    init = np.eye(4)
    init[:3, 3] = translation


    result = o3d.pipelines.registration.registration_icp(
        pcd_src, pcd_dst,
        max_correspondence_distance=max_correspondence_distance,
        init=init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    T = result.transformation
    src_pts_h = np.concatenate([src_np, np.ones((src_np.shape[0], 1))], axis=1)
    src_pts_transformed = (T @ src_pts_h.T).T[:, :3]
    flow = torch.from_numpy(src_pts_transformed - src_np).to(src_pts.device, dtype=src_pts.dtype)
    return flow

def match_instance_id(point_src, point_dst, label_src, label_dst,
                      max_center_dist=2.,
                      max_shape_ratio=2., max_correspondence_distance=0.5):
    valid_src = label_src > 0
    valid_dst = label_dst > 0
    ids_src = torch.unique(label_src[valid_src])
    ids_dst = torch.unique(label_dst[valid_dst])

    pairs = []
    new_id = torch.zeros_like(label_src)

    def extract_stats(points, labels, id):
        cluster = points[labels == id]
        center = cluster.mean(dim=0)
        extent = cluster.max(dim=0).values - cluster.min(dim=0).values
        return {'center': center, 'extent': extent, 'points': cluster}

    stats_src = {id.item(): extract_stats(point_src, label_src, id) for id in ids_src}
    stats_dst = {id.item(): extract_stats(point_dst, label_dst, id) for id in ids_dst}

    for id_src, stat_src in stats_src.items():
        best_match = None
        best_score = float('inf')
        for id_dst, stat_dst in stats_dst.items():
            center_dist = torch.norm(stat_src['center'] - stat_dst['center'])
            if center_dist > max_center_dist:
                continue
            ratio = stat_src['extent'] / (stat_dst['extent'] + 1e-6)
            ratio = torch.maximum(ratio, 1.0 / ratio)
            if torch.any(ratio > max_shape_ratio):
                continue
            score = center_dist + torch.sum(ratio).item()
            if score < best_score:
                best_score = score
                best_match = id_dst
        if best_match is not None:
            pairs.append((id_src, best_match))


    flow_3d = torch.zeros_like(point_src)

    for id_src, id_dst in pairs:
        src_mask = label_src == id_src
        dst_mask = label_dst == id_dst
        new_id[src_mask] = id_dst

        src_pts = point_src[src_mask]
        dst_pts = point_dst[dst_mask]

        if src_pts.shape[0] == 0 or dst_pts.shape[0] == 0:
            continue

        flow = icp_flow_open3d(src_pts, dst_pts, max_correspondence_distance=max_correspondence_distance)
        src_pts_orig = point_src[src_mask]
        nbrs = NearestNeighbors(n_neighbors=1).fit(src_pts.cpu().numpy())
        _, indices = nbrs.kneighbors(src_pts_orig.cpu().numpy())
        flow_selected = flow[indices[:, 0]]
        flow_3d[src_mask] = flow_selected.to(point_src.device)

    return flow_3d, pairs, new_id


def visualize_flowed_point_clouds(point_src, point_dst, flow_3d=None, num_arrows=200, window_name="Flow Visualization"):
    point_src_np = point_src.detach().cpu().numpy()
    point_dst_np = point_dst.detach().cpu().numpy()

    pcd_src = o3d.geometry.PointCloud()
    pcd_dst = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(point_src_np)
    pcd_dst.points = o3d.utility.Vector3dVector(point_dst_np)
    pcd_src.paint_uniform_color([0.1, 1.0, 0.1])  # green
    pcd_dst.paint_uniform_color([0.2, 0.6, 1.0])  # blue

    geometries = [pcd_src, pcd_dst]

    if flow_3d is not None:
        flow_np = flow_3d.detach().cpu().numpy()
        point_src_warped = point_src_np + flow_np

        pcd_warped = o3d.geometry.PointCloud()
        pcd_warped.points = o3d.utility.Vector3dVector(point_src_warped)
        pcd_warped.paint_uniform_color([1.0, 0.3, 0.3])  # red
        geometries.append(pcd_warped)

        N = point_src_np.shape[0]
        idx = np.random.choice(N, min(num_arrows, N), replace=False)
        points = []
        lines = []
        colors = []

        for i, k in enumerate(idx):
            p0 = point_src_np[k]
            p1 = point_src_np[k] + flow_np[k]
            points.append(p0)
            points.append(p1)
            lines.append([2 * i, 2 * i + 1])
            colors.append([0.0, 1.0, 0.0])  # green lines

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for g in geometries:
        vis.add_geometry(g)
    vis.run()
    vis.destroy_window()



class GaussianModel:

    @torch.no_grad()
    def fusion(self, K, sigma_c=1, sigma_s=1):
        if not self.use_fusion:
            return
        self._fusion_weighted(K, sigma_c, sigma_s)            
    
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

    def _setup_functions_base(self):
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

    def __init__(self, optimizer_type="default", use_fusion=False):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.use_fusion = use_fusion
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
        if not isinstance(self._xyz, nn.Parameter):
            self._xyz = nn.Parameter(self._xyz.detach().requires_grad_(True))
        if not isinstance(self._scaling, nn.Parameter):
            self._scaling = nn.Parameter(self._scaling.detach().requires_grad_(True))

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init, "name": "xyz"},
            # {'params': [self._rgb], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
        self.optimizer.zero_grad(set_to_none=True)

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

        chunk_size = 5000
        if point_offset is not None:
            xyz_ls = []
            sem_ls = []
            for chunk in range(0, point_xyz.shape[0], chunk_size):
                point_xyz_chunk = point_xyz[chunk:chunk+chunk_size]
                point_sem_chunk = point_sem[chunk:chunk+chunk_size]
                point_radius_chunk = point_radius[chunk:chunk+chunk_size]
                point_rotation_chunk = point_rotation[chunk:chunk+chunk_size]

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

        return voxel_indices, voxel_cls

    def setup_functions(self):
        self._setup_functions_base()
        self.scaling_activation = torch.sigmoid
        self.scaling_inverse_activation = inverse_sigmoid

    @torch.no_grad()
    def create_from_pcd(self, train_data):
        self.world2cam_ls_prev = [
            torch.tensor(train_data['world2cam_ls'][cam], device='cuda').float()
            for cam in range(len(train_data['world2cam_ls']))
        ]
        if "point_instance_id" in train_data and not hasattr(self, 'point_instance_id_src'):
            self.point_instance_id_src = train_data['point_instance_id']
        N = train_data['point_xyz'].shape[0]
        self._xyz = train_data['point_xyz']
        self._rgb = train_data['point_rgb']
        self._opacity = self.inverse_opacity_activation(torch.full((N, 1), 0.5, device='cuda'))
        self._semantic = train_data['point_sem']
        dist2 = torch.clamp_min(distCUDA2(train_data['point_xyz']), 0.0000001)
        scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)
        self._scaling = scales
        rots = torch.zeros((N, 4), device="cuda")
        rots[:, 0] = 1
        self._rotation = rots
        if 'nonground_mask' in train_data.keys():
            self.nonground_mask = train_data['nonground_mask']
        self.world2cam_ls = train_data['world2cam_ls']
        scales = self.scaling_inverse_activation(torch.full((self.N, 3), 0.3, device='cuda'))
        self._scaling = scales

    def add_from_pcd(self, train_data, *args, **kwargs):
        if "dynamic_mask_ls" in train_data:
            xyz_world_prev = self.detach_dict['_xyz']
            del_dyn_point_mask = torch.zeros(len(xyz_world_prev), dtype=torch.bool, device=xyz_world_prev.device)
            for cam in range(6):
                dynamic_mask = train_data['dynamic_mask_ls'][cam]
                H, W = dynamic_mask.shape

                world2cam_prev = self.world2cam_ls_prev[cam]
                intrinsic = train_data['intrinsic3x3_ls'][cam]
                uv = project_to_pixel(xyz_world_prev, world2cam_prev, intrinsic)

                u = uv[0].round().long()
                v = uv[1].round().long()
                in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                del_dyn_point_mask[in_bounds] |= dynamic_mask[v[in_bounds], u[in_bounds]]
            


        elif "point_instance_id" in train_data:
            point_instance_id_dst = train_data['point_instance_id']
            start_time = time.time()
            flow_3d, pairs, new_id = match_instance_id( 
                           point_src=self.detach_dict['_xyz'], 
                           point_dst=train_data['point_xyz'], 
                           label_src=self.point_instance_id_src,
                           label_dst=point_instance_id_dst
                        )
            # print(f"Matching instance IDs took {(time.time() - start_time)*1000:.0f} ms")
            
            # for pair in pairs:
            #     visualize_flowed_point_clouds(self.detach_dict['_xyz'][self.point_instance_id_src == pair[0]], 
            #                                   train_data['point_xyz'][point_instance_id_dst == pair[1]], 
            #                                   flow_3d[self.point_instance_id_src == pair[0]], window_name=f"Flow Visualization for ID {pair[0]} to {pair[1]}")
            self.detach_dict['_xyz'] += flow_3d
            del_dyn_point_mask = (self.point_instance_id_src > 0) & (new_id == 0)
            new_id = new_id[~del_dyn_point_mask]
            self.point_instance_id_src = torch.cat([point_instance_id_dst, new_id], dim=0)

        else:
            raise ValueError("train_data must contain dynamic_mask_ls or point_instance_id")

        self.detach_dict['_xyz'] = self.detach_dict['_xyz'][~del_dyn_point_mask]
        self.detach_dict['_rgb'] = self.detach_dict['_rgb'][~del_dyn_point_mask]
        self.detach_dict['_opacity'] = self.detach_dict['_opacity'][~del_dyn_point_mask]
        self.detach_dict['_scaling'] = self.detach_dict['_scaling'][~del_dyn_point_mask]
        self.detach_dict['_rotation'] = self.detach_dict['_rotation'][~del_dyn_point_mask]
        self.detach_dict['_semantic'] = self.detach_dict['_semantic'][~del_dyn_point_mask]
        self.create_from_pcd(train_data)
        self._xyz = torch.cat((self._xyz, self.detach_dict['_xyz']), dim=0)
        self._rgb = torch.cat((self._rgb, self.detach_dict['_rgb']), dim=0)
        self._opacity = torch.cat((self._opacity, self.detach_dict['_opacity']), dim=0)
        self._scaling = torch.cat((self._scaling, self.detach_dict['_scaling']), dim=0)
        self._rotation = torch.cat((self._rotation, self.detach_dict['_rotation']), dim=0)
        self._semantic = torch.cat((self._semantic, self.detach_dict['_semantic']), dim=0)
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
    def _fusion_weighted(self, K, sigma_c=1, sigma_s=1):
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