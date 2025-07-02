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

from arguments import colors, num_classes, empty_id, settings, colors_1
from scipy.spatial import cKDTree

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
        
        if dataset is not None:
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

def sparse2dense(voxel_indices, voxel_cls, min_bound, max_bound, voxel_size):
    # print('sparse2dense')
    dense_size = ((max_bound-min_bound) / voxel_size + 1e-6)[0].astype(int).tolist()
    dense_cls = np.zeros((dense_size), dtype=int)+empty_id
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