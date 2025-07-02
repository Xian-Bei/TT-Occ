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

import numpy as np
import torch
from torch import nn
from scene.gaussian_model_C_add_hist import GaussianModel as GaussianModel_Parent
import ipdb
import time
import torch.nn.functional as F
import hdbscan
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import hdbscan
from scene.dataset_readers import visualize_point_clouds
from custom_utils import project_to_pixel

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


class GaussianModel(GaussianModel_Parent):
    @torch.no_grad()
    def create_from_pcd(self, train_data):
        self.world2cam_ls_prev = [torch.tensor(train_data['world2cam_ls'][cam], device='cuda').float() for cam in range(len(train_data['world2cam_ls']))]
        if "point_instance_id" in train_data and not hasattr(self, 'point_instance_id_src'):
            self.point_instance_id_src = train_data['point_instance_id']
        super().create_from_pcd(train_data)
        

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
            raise
        
        # print(f"dynamic ratio: {del_dyn_point_mask.sum()/len(del_dyn_point_mask): .2%}")
        self.detach_dict['_xyz'] = self.detach_dict['_xyz'][~del_dyn_point_mask]
        self.detach_dict['_rgb'] = self.detach_dict['_rgb'][~del_dyn_point_mask]
        self.detach_dict['_opacity'] = self.detach_dict['_opacity'][~del_dyn_point_mask]
        self.detach_dict['_scaling'] = self.detach_dict['_scaling'][~del_dyn_point_mask]
        self.detach_dict['_rotation'] = self.detach_dict['_rotation'][~del_dyn_point_mask]
        self.detach_dict['_semantic'] = self.detach_dict['_semantic'][~del_dyn_point_mask]
        super().add_from_pcd(train_data, *args, **kwargs)
        # visualize_point_clouds(self._xyz[self.point_instance_id_src>0], self.point_instance_id_src[self.point_instance_id_src>0], 'after add_from_pcd')