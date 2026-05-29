import math
import torch
import ipdb
import open3d as o3d
    
import os
import shutil
import PIL
from colmap_loader import *
from utils.graphics_utils import focal2fov
import numpy as np
import cv2
import joblib
from os.path import join
from utils.camera_utils import CameraInfo
import gc
from utils.ray_metrics_cuda import process_one_sample, calc_rayiou, PrettyTable
from arguments import colors, num_classes, empty_id, settings, colors_1
from scipy.spatial import cKDTree

import numpy as np

def transform_occ_from_ego_to_lidar(
    voxel_indices,
    voxel_cls,
    ego2lidar,
    pc_range_input=np.array([-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]),
    pc_range_output=np.array([-51.2, -51.2, -5, 51.2, 51.2, 3]),
    voxel_size=0.2,
    fill_value=255
):
    """
    Transform sparse voxel representation from ego to LiDAR coordinate system,
    supporting different input/output pc_range sizes.

    Args:
        voxel_indices: (N, 3) numpy array of voxel indices in ego coordinates.
        voxel_cls: (N,) numpy array of class labels corresponding to voxel_indices.
        ego2lidar: (4, 4) numpy array, transform from ego to LiDAR coordinates.
        pc_range_input: array-like (6,), input voxel grid range [x_min, y_min, z_min, x_max, y_max, z_max].
        pc_range_output: array-like (6,), desired output grid range.
        voxel_size: float, resolution of the voxel grid.
        fill_value: int, value to fill in unmapped voxels.

    Returns:
        transformed_grid: (H_out, W_out, Z_out) numpy array in LiDAR coordinate.
    """

    # 1. 将索引转换为 ego 坐标系下的三维点
    points = voxel_indices * voxel_size + pc_range_input[:3][None]  # [N, 3]

    # 2. 转换为 lidar 坐标系
    points_homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)  # [N, 4]
    points_lidar = (ego2lidar @ points_homo.T).T[:, :3]  # [N, 3]

    valid_mask = (points_lidar >= pc_range_output[:3]).all(1) & (points_lidar < pc_range_output[3:]).all(1)
    points_lidar = points_lidar[valid_mask]
    voxel_cls = voxel_cls[valid_mask]
    points_lidar = np.floor((points_lidar - pc_range_output[:3]) / voxel_size).astype(int)
    return sparse2dense(points_lidar, voxel_cls, pc_range_output[:3][None], pc_range_output[3:][None], voxel_size, fill_value=fill_value)


@torch.no_grad()
def transform_occ_from_ego_to_lidar_torch(
    voxel_indices: torch.Tensor,          # (N, 3) long/int on CUDA/CPU
    voxel_cls: torch.Tensor,              # (N,) labels on same device
    ego2lidar: torch.Tensor,              # (4, 4) float on same device
    pc_range_input=( -40.0, -40.0, -1.0,  40.0,  40.0, 5.4),
    pc_range_output=(-51.2, -51.2, -5.0,  51.2,  51.2, 3.0),
    voxel_size: float = 0.2,
    fill_value: int = 255,
):
    """
    将稀疏体素 (ego 坐标系) -> LiDAR 坐标系 -> 在输出范围下稠密化为 (H_out, W_out, Z_out)。

    约定稠密网格维度顺序为 (H, W, Z) ≡ (y, x, z)，索引写入时使用 [y, x, z]。
    最终张量 dtype 使用 voxel_cls 的 dtype；未命中体素填 fill_value。
    """

    device = voxel_indices.device
    dtype_lab = voxel_cls.dtype

    # ---- 0) 参数张量化 ----
    pc_in  = torch.tensor(pc_range_input,  device=device, dtype=torch.float32)
    pc_out = torch.tensor(pc_range_output, device=device, dtype=torch.float32)
    out_min = pc_out[:3]   # (x_min, y_min, z_min)
    out_max = pc_out[3:]   # (x_max, y_max, z_max)

    # ---- 1) index -> ego 坐标 ----
    # points_ego = min_input + idx * voxel_size
    points_ego = voxel_indices.to(torch.float32) * float(voxel_size) + pc_in[:3]  # (N, 3)

    # ---- 2) ego -> lidar ----
    ones = torch.ones((points_ego.shape[0], 1), device=device, dtype=torch.float32)
    points_h = torch.cat([points_ego, ones], dim=1)                 # (N, 4)
    # 注意右乘 or 左乘：这里按 (N,4) @ (4,4)^T
    points_lidar = (points_h @ ego2lidar.transpose(0, 1))[:, :3]    # (N, 3)

    # ---- 3) 过滤落在输出范围内的点 ----
    valid = (points_lidar >= out_min) & (points_lidar < out_max)    # (N, 3) bool
    valid = valid.all(dim=1)                                        # (N,)
    if valid.sum() == 0:
        # 构造空网格直接返回
        W_out = int(torch.round((out_max[0] - out_min[0]) / voxel_size).item())
        H_out = int(torch.round((out_max[1] - out_min[1]) / voxel_size).item())
        Z_out = int(torch.round((out_max[2] - out_min[2]) / voxel_size).item())
        grid = torch.full((H_out, W_out, Z_out), fill_value=fill_value, dtype=dtype_lab, device=device)
        return grid

    points_lidar = points_lidar[valid]
    voxel_cls    = voxel_cls[valid]

    # ---- 4) 输出网格大小 (H_out, W_out, Z_out) ----
    # 注意维度对应：x->W, y->H, z->Z
    sizes = torch.round((out_max - out_min) / voxel_size).to(torch.long)  # (3,)
    W_out, H_out, Z_out = int(sizes[0].item()), int(sizes[1].item()), int(sizes[2].item())

    # ---- 5) lidar 坐标 -> 输出网格索引 (x,y,z) -> (y,x,z) ----
    idx_float = (points_lidar - out_min) / voxel_size                         # (N,3)
    idx_xyz   = torch.floor(idx_float).to(torch.long)                         # (N,3)
    x_idx, y_idx, z_idx = idx_xyz[:, 0], idx_xyz[:, 1], idx_xyz[:, 2]

    # 再次严格越界保护（浮点误差安全网）
    in_bounds = (
        (x_idx >= 0) & (x_idx < W_out) &
        (y_idx >= 0) & (y_idx < H_out) &
        (z_idx >= 0) & (z_idx < Z_out)
    )
    if in_bounds.sum() == 0:
        grid = torch.full((H_out, W_out, Z_out), fill_value=fill_value, dtype=dtype_lab, device=device)
        return grid

    x_idx, y_idx, z_idx = x_idx[in_bounds], y_idx[in_bounds], z_idx[in_bounds]
    voxel_cls = voxel_cls[in_bounds]

    # ---- 6) 稠密化：按 [y,x,z] 写入 ----
    grid = torch.full((H_out, W_out, Z_out), fill_value=fill_value, dtype=dtype_lab, device=device)

    # 将 3D 索引展平，使用 index_copy 写入（重复位置后出现者覆盖先前者）
    flat_idx = (y_idx * (W_out * Z_out)) + (x_idx * Z_out) + z_idx          # (M,)
    grid_flat = grid.view(-1)
    grid_flat.index_copy_(0, flat_idx, voxel_cls.to(dtype_lab))
    grid = grid_flat.view(H_out, W_out, Z_out)

    return grid


def format_floats_as_percentage(d):
    for key, value in d.items():
        if isinstance(value, float):
            d[key] = f"{value * 100:.2f}%"
        elif isinstance(value, dict):
            format_floats_as_percentage(value)  
    return d

def eval_occ(timestep, voxel_indices_from_gs, voxel_cls_from_gs, occ_setting, scene, mapping, gt_path=None):
    dense_cls_from_gs = sparse2dense_torch(voxel_indices_from_gs, voxel_cls_from_gs, *settings[occ_setting])
    if occ_setting == "Occ3D":
        dense_cls_occ_np, mask = load_occ3d_gt(
            join(gt_path, scene, f"{mapping['sample_token'][f'{timestep:0>2d}']}/labels.npz"))
        dense_cls_occ = torch.tensor(dense_cls_occ_np, device='cuda')
        mask = torch.tensor(mask, device='cuda')
        
        new_hist_occ_camera = cal_hist(pred_occ=dense_cls_from_gs[mask], gt_occ=dense_cls_occ[mask])

        return new_hist_occ_camera, dense_cls_occ_np
        
    elif occ_setting == "nuCraft":
        nucraft_gt = load_nucraft_gt(join(gt_path, f"{mapping['LIDAR_TOP'][f'{timestep:0>2d}']}.bin"))
        dense_cls_nucraft = torch.tensor(sparse2dense(*nucraft_gt, *settings['nuCraft_np']), device='cuda')
        new_hist_nu = cal_hist(dense_cls_from_gs, dense_cls_nucraft)
        return new_hist_nu, nucraft_gt
        
        
    else:
        raise ValueError("Unsupported occ_setting")

def eval_occ_2(timestep, dense_cls_from_gs, occ_setting, scene, mapping, gt_path=None, cnm_mask=None):
    nucraft_gt = load_nucraft_gt(join(gt_path, f"{mapping['LIDAR_TOP'][f'{timestep:0>2d}']}.bin"))
    dense_cls_nucraft = torch.tensor(sparse2dense(*nucraft_gt, *settings['nuCraft_np']), device='cuda')
    if cnm_mask is not None:
        dense_cls_nucraft[cnm_mask] = 17
    new_hist_nu = cal_hist(dense_cls_from_gs, dense_cls_nucraft)
    return new_hist_nu, nucraft_gt
    
def process_inverse_depth_map(inverse, epsilon=1e-8, threshold=100): # image (1, H, W)
    inverse = inverse.detach().cpu().squeeze(0) # (H, W)
    inverse = torch.clamp_min(inverse, min=1/threshold)
    depth = 1.0 / inverse
    # depth[depth>threshold] = 0
    return depth
    
def custom_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    if isinstance(obj, float) and np.isnan(obj): 
        return None  
    return obj
    
def query_np_kdtree(query_np, tree_np, k, max_dist):
    tree = cKDTree(tree_np)
    return tree.query(query_np, k=k, workers=-1, distance_upper_bound=max_dist)

class VoxelGridVisualizer:
    def __init__(self, name='', num=1, width=200, dataset='Occ3D'):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.width = width
        self.vis.create_window(window_name=name, width=width*num, height=width, visible=True, left=0, top=0)
        self.vis.get_render_option().background_color = np.array([1, 1, 1])
        self.continue_requested = False  
        # self.view_control = self.vis.get_view_control()
        self.select_idx = 0
        self.vis.register_key_callback(ord("C"), self.request_continue)
        self.vis.register_key_callback(ord("Q"), self.request_exit)
        self.vis.register_key_callback(ord("S"), self.save)
        self.vis.register_key_callback(ord("P"), self.print_ext)
        self.vis.register_key_callback(ord("1"), self.set1)
        self.vis.register_key_callback(ord("2"), self.set2)
        self.vis.register_key_callback(ord("3"), self.set3)
        self.vis.register_key_callback(ord("4"), self.set4)
        self.vis.register_key_callback(ord("5"), self.set5)
        self.vis.register_key_callback(ord("6"), self.set6)
        self.vis.register_key_callback(ord("7"), self.set7)
        self.vis.register_key_callback(ord("8"), self.set8)
        self.vis.register_key_callback(ord("9"), self.set9)
        self.vis.register_key_callback(ord("0"), self.set10)
        
        if dataset == 'nuCraft':
            self.camera_params = o3d.camera.PinholeCameraParameters()
            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            intrinsic.set_intrinsics(width, width, 519.61524227, 519.61524227, 299.5, 299.5)
            self.camera_params.intrinsic = intrinsic
        
            extrinsic = np.array([   [ 9.96852397e-01,  7.61305499e-02, -2.21232519e-02, -2.59219146e+02],
                [ 3.67059843e-03, -3.23073402e-01, -9.46366791e-01,  1.12103288e+02],
                [-7.91948585e-02,  9.43306799e-01, -3.22335940e-01, -2.62678231e+01],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],])
            self.camera_params.extrinsic = extrinsic
        
            self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.camera_params)
    
    def reset(self, voxel_grid_ls, image_path=None):
        self.voxel_grid_ls = voxel_grid_ls
        self.image_path = image_path
        
    def set(self, idx, continue_requested=False):
        if idx < len(self.voxel_grid_ls):
            self.select_idx = idx
        self.update_voxel_grid(continue_requested=continue_requested)

    def set1(self, vis):
        self.set(0)
    def set2(self, vis):
        self.set(1)
    def set3(self, vis):
        self.set(2)
    def set4(self, vis):    
        self.set(3)
    def set5(self, vis):
        self.set(4)
    def set6(self, vis):
        self.set(5)
    def set7(self, vis):
        self.set(6)
    def set8(self, vis):
        self.set(7)
    def set9(self, vis):
        self.set(8)
    def set10(self, vis):
        self.set(9)
        
    def print_ext(self, vis):
        print("extrinsic", self.camera_params.extrinsic)
        
    def request_continue(self, vis):
        self.continue_requested = True
    
    def request_exit(self, vis):
        self.vis.close()
        self.vis.destroy_window()
        
    def save(self, vis):
        # self.continue_requested = True
        self.vis.capture_screen_image(self.image_path+f"_{self.select_idx}.png")
        print(f"Saved image to {self.image_path+f'_{self.select_idx}.png'}")

    def update_voxel_grid(self, continue_requested=False):
        ############################
        self.continue_requested = continue_requested
        ############################
        self.camera_params = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
        
        self.vis.clear_geometries() 
        self.vis.add_geometry(self.voxel_grid_ls[self.select_idx])
        self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.camera_params)
        self.vis.poll_events()
        self.vis.update_renderer()
        
        while not self.continue_requested:
            self.vis.poll_events()  
            self.vis.update_renderer()

def view_sparse_voxel(view_ls, vis, path=None):
    voxel_grid_ls = []
    for i, (voxel_indices, voxel_cls) in enumerate(view_ls):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(voxel_indices+0.5)
        pcd.colors = o3d.utility.Vector3dVector(colors_1[voxel_cls])  # Normalize colors to [0, 1]
        voxel_grid_ls.append(o3d.geometry.VoxelGrid.create_from_point_cloud(input=pcd, voxel_size=1))
    vis.reset(voxel_grid_ls, path)
    vis.update_voxel_grid()

def sparse2dense(voxel_indices, voxel_cls, min_bound, max_bound, voxel_size, fill_value=empty_id):
    # print('sparse2dense')
    dense_size = ((max_bound-min_bound) / voxel_size + 1e-6)[0].astype(int).tolist()
    dense_cls = np.full((dense_size), fill_value, dtype=int)
    dense_cls[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = voxel_cls
    return dense_cls

def sparse2dense_torch(voxel_indices, voxel_cls, min_bound, max_bound, voxel_size):
    # print('sparse2dense_torch')
    dense_size = ((max_bound-min_bound) / voxel_size + 1e-6).int()[0].tolist()
    dense_cls = torch.full((dense_size), empty_id, device=voxel_indices.device)
    dense_cls[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = voxel_cls
    return dense_cls
        
def dense2sparse(dense_cls):
    voxel_indices = np.stack(np.nonzero(dense_cls!=empty_id), axis=-1)
    voxel_cls = dense_cls[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]]
    return voxel_indices, voxel_cls

def load_nucraft_gt(file):
    id_mapping = np.array([
        0, # 0->0 noise
        0, # 1->0 animal
        7, # 2->7 human.pedestrian.adult
        7, # 3->7 human.pedestrian.child
        7, # 4->7 human.pedestrian.construction_worker
        0, # 5->0 human.pedestrian.personal_mobility
        7, # 6->7 human.pedestrian.police_officer
        0, # 7->0 human.pedestrian.stroller
        0, # 8->0 human.pedestrian.wheelchair
        1, # 9->1 movable_object.barrier
        0, # 10->0 movable_object.debris
        0, # 11->0 movable_object.pushable_pullable
        8, # 12->8 movable_object.trafficcone
        0, # 13->0 static_object.bicycle_rack
        2, # 14->2 vehicle.bicycle
        3, # 15->3 vehicle.bus.bendy
        3, # 16->3 vehicle.bus.rigid
        4, # 17->4 vehicle.car
        5, # 18->5 vehicle.construction
        0, # 19->0 vehicle.emergency.ambulance
        0, # 20->0 vehicle.emergency.police
        6, # 21->6 vehicle.motorcycle
        9, # 22->9 vehicle.trailer
        10, # 23->10 vehicle.truck
        11, # 24->11 flat.driveable_surface
        12, # 25->12 flat.other
        13, # 26->13 flat.sidewalk
        14, # 27->14 flat.terrain
        15, # 28->15 static.manmade
        0, # 29->0 static.other
        16, # 30->16 static.vegetation
        0, # 31->0 vehicle.ego
    ])
    # print('load_nucraft_gt')
    data = np.fromfile(file, dtype=np.int16).reshape(-1, 4)
    voxel_indices = np.empty((data.shape[0], 3), dtype=np.int16)
    voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2] = data[:, 2], data[:, 1], data[:, 0] # X, Y, Z: 0-1023, 0-1023, 0-79
    # voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2] = data[:, 0], data[:, 1], data[:, 2] # X, Y, Z: 0-1023, 0-1023, 0-79
    voxel_cls = id_mapping[data[:, -1]] # 0-31
    return voxel_indices, voxel_cls

def load_occ3d_gt(file):
    # print('load_occ3d_gt')
    data = np.load(file)
    dense_cls = data['semantics'].astype(int)
    mask = data['mask_camera'].astype(bool)
    return dense_cls, mask

def cal_hist(pred_occ, gt_occ):
    if isinstance(pred_occ, torch.Tensor):
        assert isinstance(gt_occ, torch.Tensor)
        return torch.bincount(
            (num_classes+1) * gt_occ.flatten() + pred_occ.flatten(), # row: gt, col: pred
            minlength=(num_classes+1)**2
        ).view((num_classes+1), (num_classes+1))
    else:
        assert isinstance(gt_occ, np.ndarray)
        return np.bincount(
            (num_classes+1) * gt_occ.flatten().astype(int) + pred_occ.flatten(),
            minlength=(num_classes+1)**2
        ).reshape((num_classes+1), (num_classes+1))
    
def cal_iou_miou(hist):
    # not consider 0 (noise) and 12 (other_flat) class
    if isinstance(hist, torch.Tensor):
        hist = torch.cat((hist[1:12], hist[13:]), dim=0)  
        hist = torch.cat((hist[:, 1:12], hist[:, 13:]), dim=1) 
        n = hist.shape[0]

        TP_occupied = torch.sum(hist[:n-1, :n-1])  
        FP_occupied = torch.sum(hist[n-1, :n-1])   
        FN_occupied = torch.sum(hist[:n-1, n-1])   

        iou = TP_occupied / (TP_occupied + FP_occupied + FN_occupied)

        # hist = hist[:n-1, :n-1]

        intersection = torch.diag(hist)
        mious = intersection / (hist.sum(dim=1) + hist.sum(dim=0) - intersection)
        mious = mious[:-1]  

        miou = torch.nanmean(mious)
    else:
        ipdb.set_trace()
        hist = np.delete(hist, [0, 12], axis=0)
        hist = np.delete(hist, [0, 12], axis=1)
        
        n = hist.shape[0]
        TP_occupied = np.sum(hist[:n-1, :n-1])  
        FP_occupied = np.sum(hist[n-1, :n-1])   
        FN_occupied = np.sum(hist[:n-1, n-1])   
        iou = TP_occupied / (TP_occupied + FP_occupied + FN_occupied)
        
        # hist = hist[:n-1, :n-1]

        intersection = np.diag(hist)
        mious = intersection / (hist.sum(1) + hist.sum(0) - intersection)
        mious = mious[:-1]

        miou = np.nanmean(mious)
    
    return iou, miou, mious

def unproject_to_world(x, y, depth, K, cam2world):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (x - cx) * depth / fx
    y = (y - cy) * depth / fy
    z = depth
    pts_cam = np.stack([x, y, z], axis=1)  # (N, 3)

    pts_cam_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1))])  # (N, 4)
    pts_world = (cam2world @ pts_cam_h.T).T[:, :3]  # (N, 3)
    return pts_world

def project_to_pixel(xyz_world, world2cam, K):
    N = xyz_world.shape[0]
    if isinstance(xyz_world, torch.Tensor):
        xyz_h = torch.cat([xyz_world, torch.ones((N, 1), device=xyz_world.device, dtype=xyz_world.dtype)], dim=1)
    else:
        xyz_h = np.hstack([xyz_world, np.ones((N, 1), dtype=xyz_world.dtype)])

    xyz_cam = (world2cam @ xyz_h.T).T[:, :3]
    x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
    valid_mask = z > 0
    z[~valid_mask] = 1e-6  # Avoid division by zero
    u = (K[0, 0] * x / z) + K[0, 2]
    v = (K[1, 1] * y / z) + K[1, 2]
    return u, v, valid_mask