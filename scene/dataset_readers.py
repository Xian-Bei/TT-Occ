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

import os
import time
from typing import NamedTuple

from matplotlib import pyplot as plt
from colmap_loader import *
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import cv2
from os.path import join
from utils.camera_utils import loadCam
from arguments import ego_range, num_classes
import open3d as o3d
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from custom_utils import unproject_to_world, project_to_pixel
from submodules.vggt.main import VGGT, process_frame_vggt
from submodules.RAFT.main import process_frame_raft, load_args

def visualize_point_clouds(pts, ids, window_name="Point Cloud Visualization"):
    if isinstance(pts, torch.Tensor):
        pts = pts.cpu().numpy()
    if isinstance(ids, torch.Tensor):
        ids = ids.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    unique_ids = np.unique(ids)
    unique_ids = unique_ids[unique_ids != 0] 
    cmap = plt.get_cmap("tab20", len(unique_ids))

    color_map = {0: [0.5, 0.5, 0.5]}  
    for idx, label in enumerate(unique_ids):
        color = cmap(idx % 20)[:3]
        color_map[label] = color

    colors = np.array([color_map[int(i)] for i in ids], dtype=np.float32)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, )
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    
from sklearn.cluster import DBSCAN
def denoise_instance_cluster(xyz, vis=False):
    N = xyz.shape[0]
    dbscan = DBSCAN(eps=1)
    labels = dbscan.fit_predict(xyz)

    valid_mask = labels != -1
    if np.sum(valid_mask) == 0:
        return None 

    inlier_mask = np.zeros(N, dtype=bool)
    labels_filtered = labels[valid_mask]
    largest_label = np.bincount(labels_filtered).argmax()

    indices_valid = np.where(valid_mask)[0]
    final_indices = indices_valid[labels_filtered == largest_label]
    inlier_mask[final_indices] = True


    if vis:
        pcd_original = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(xyz)
        pcd_original.paint_uniform_color([0.5, 0.5, 0.5]) 

        filtered_xyz = xyz[inlier_mask]
        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(filtered_xyz)
        pcd_filtered.paint_uniform_color([0.0, 1.0, 0.0])

        pcd_filtered.translate((2, 0, 0))  
        o3d.visualization.draw_geometries([pcd_original, pcd_filtered])

    return inlier_mask
    
def voxelize_torch(point_xyz, point_rgb, point_sem, voxel_size=0.1):
    device = point_xyz.device
    voxel_indices = torch.floor(point_xyz / voxel_size).to(torch.int32)  # (N, 3)

    voxel_indices_flat = voxel_indices.view(-1, 3)
    unique_voxels, inverse_indices, counts = torch.unique(voxel_indices_flat, dim=0, return_inverse=True, return_counts=True)

    voxel_xyz = (unique_voxels.to(torch.float32) + 0.5) * voxel_size

    num_voxels = unique_voxels.shape[0]
    num_classes = point_sem.shape[1]

    accumulated_rgb = torch.zeros((num_voxels, 3), dtype=torch.float32, device=device)
    accumulated_sem = torch.zeros((num_voxels, num_classes), dtype=torch.float32, device=device)

    accumulated_rgb = accumulated_rgb.index_add(0, inverse_indices, point_rgb)
    accumulated_sem = accumulated_sem.index_add(0, inverse_indices, point_sem)

    voxel_rgb = accumulated_rgb / counts.unsqueeze(1)

    voxel_sem = accumulated_sem + 1e-6
    voxel_sem = voxel_sem / voxel_sem.sum(dim=1, keepdim=True)

    return voxel_xyz, voxel_rgb, voxel_sem

def voxelize(point_xyz, point_rgb, point_sem, voxel_size=0.1): # point_sem in shape (N, num_classes)

    voxel_indices = np.floor(point_xyz / voxel_size).astype(int)
    unique, unique_inverse, unique_counts = np.unique(voxel_indices, axis=0, return_inverse=True, return_counts=True)
    point_xyz = (unique+0.5) * voxel_size

    # downsampled_points = np.zeros((len(unique), 3), dtype=np.float32)
    accumulated_rgb = np.zeros((len(unique), 3), dtype=np.float32)
    accumulated_sem = np.zeros((len(unique), point_sem.shape[1]), dtype=np.float32)
    
    np.add.at(accumulated_rgb, unique_inverse, point_rgb)
    np.add.at(accumulated_sem, unique_inverse, point_sem)
    '''np.add.at =
    for i, id in enumerate(unique_inverse):
        downsampled_colors[id] += point_rgb[i]
    '''
    point_rgb = accumulated_rgb / unique_counts[:, None]
    point_sem = accumulated_sem
    
    point_sem += 1e-6
    point_sem = point_sem / point_sem.sum(1, keepdims=True)
    return point_xyz, point_rgb, point_sem

def remove_radius_outlier_numpy(points: np.ndarray, radius: float, min_neighbors: int) -> np.ndarray:
    """
    Remove radius outliers from point cloud using a numpy + scipy-based KDTree.
    
    Args:
        points (N, 3): Input point cloud.
        radius (float): Radius within which to search for neighbors.
        min_neighbors (int): Minimum number of neighbors required to keep a point.

    Returns:
        mask (N,): Boolean mask where True indicates the point is kept.
    """
    tree = cKDTree(points)
    counts = tree.query_ball_point(points, r=radius, return_length=True)
    mask = counts >= min_neighbors
    return mask

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool
    semantic_path: str = None
    intrinsic: list = None

class SceneInfo(NamedTuple):
    # point_cloud: BasicPointCloud
    train_data: dict
    train_cameras: list
    test_cameras: list
    world2ego: np.array
    world2lid: np.array
    # nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool


vggt_model, intrinsic3x3_scaled_vggt, prev_scale = None, None, 20
raft_model, raft_padder, image_prev, depth_prev, cam2world_prev, intrinsic3x3_scaled_raft = None, None, None, None, None, None
def load_nuscenes(path, frame, old_world2new_world, intrinsic3x3_ls, source='lidar'):
    sem_prefix = "openseed"
    if source == 'lidar':
        lidar_dir = join(path, "lidar")
        ego_dir = join(path, "ego")
        train_cam_infos = []

        lidar2world = np.loadtxt(join(lidar_dir, f"{frame:0>2d}.txt"))
        lidar2world = old_world2new_world @ lidar2world
        world2lidar = np.linalg.inv(lidar2world)
        
        ego2world = np.loadtxt(join(ego_dir, f"{frame:0>2d}.txt"))
        ego2world = old_world2new_world @ ego2world
        world2ego = np.linalg.inv(ego2world)

        xyz_lidar = np.fromfile(join(lidar_dir, f"{frame:0>2d}.bin"), dtype=np.float32).reshape(-1, 4)[:, :3]

        w, l, h, slack = ego_range['w'], ego_range['l'], ego_range['h'], ego_range['eps']        
        inside_ego_range = np.array([
            [-w/2-slack, -l*0.4-slack, -h-slack],
            [w/2+slack, l*0.6+slack, slack]
        ])
        inside_mask = (xyz_lidar >= inside_ego_range[0:1]).all(1) & (xyz_lidar <= inside_ego_range[1:]).all(1) 
        xyz_lidar = xyz_lidar[~inside_mask]
        
        xyz_world = (lidar2world[:3, :3] @ xyz_lidar.T + lidar2world[:3, 3:4]).T
        
        N = xyz_world.shape[0]
        point_rgb_sum = np.zeros((N, 3), dtype=np.uint8)
        point_count = np.zeros((N), dtype=np.uint8)
        point_sem = np.zeros((N, num_classes), dtype=np.uint8)
        point_instance_id = np.zeros((N), dtype=np.uint8)
        instance_mask_ls = []
        world2cam_ls = []
        for cam in range(6):
            name = f"{cam}/{frame:0>2d}"
            image_name = f"{name}.jpg"
            image_id = 0 #len(Image_ls)
            image_path = join(path, image_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            width, height = image.shape[1], image.shape[0]
            semantic_path = join(path, f"{sem_prefix}_{cam}/{frame:0>2d}.png")
            if not os.path.exists(semantic_path):
                semantic_path = ''
                semantic_image = None
            else:
                semantic_image = cv2.imread(semantic_path, -1)
                semantic_image[semantic_image == 17] = 0 # sky -> unsure

            cam2world = np.loadtxt(join(path, f"{name}.txt"))
            cam2world = old_world2new_world @ cam2world
            R_c2w = cam2world[:3, :3]
            world2cam = np.linalg.inv(cam2world)
            R_w2c = world2cam[:3, :3]
            T_w2c = world2cam[:3, 3]
            fovy = focal2fov(intrinsic3x3_ls[cam][1,1], height)
            fovx = focal2fov(intrinsic3x3_ls[cam][0,0], width)
            cam_info = CameraInfo(uid=image_id, R=R_c2w, T=T_w2c, 
                FovY=fovy, FovX=fovx,
                depth_params=None,
                image_path=image_path, image_name=image_name, 
                depth_path="", 
                semantic_path=semantic_path,
                width=width, height=height, is_test=False)
            train_cam_infos.append(cam_info)
            ###########################################################################
            xyz_camera = (R_w2c @ xyz_world.T + T_w2c.reshape(3, 1)).T
            xyz_image = (intrinsic3x3_ls[cam] @ xyz_camera.T).T
            depth = xyz_image[:, 2]
            uv_image = xyz_image[:, :2] / xyz_image[:, 2:3]
            valid_mask = (depth > 0) \
                & (uv_image[:, 0] >= 0) \
                    & (uv_image[:, 0] < width) \
                        & (uv_image[:, 1] >= 0) \
                            & (uv_image[:, 1] < height)
            uv_image = uv_image[valid_mask].astype(np.float32)

            if np.count_nonzero(valid_mask) > 0:
                point_count[valid_mask] += 1
                point_rgb_sum[valid_mask] += cv2.remap(image, uv_image[:, 0], uv_image[:, 1], interpolation=cv2.INTER_LINEAR).reshape(-1, 3)
                if semantic_image is not None:
                    instance_path = join(path, f"{sem_prefix}_{cam}/{frame:0>2d}_instance.png")
                    labels_at_projected_pixels = cv2.remap(semantic_image, uv_image[:, 0], uv_image[:, 1], interpolation=cv2.INTER_NEAREST).reshape(-1)
                    if os.path.exists(instance_path):
                        instance_mask = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED)
                        res = cv2.remap(instance_mask, uv_image[:, 0], uv_image[:, 1], interpolation=cv2.INTER_NEAREST).reshape(-1)
                        for i in range(1, res.max()+1):
                            i_mask = (res == i)
                            if i_mask.sum() == 0:
                                continue
                            inlier_mask = denoise_instance_cluster(xyz_world[valid_mask][i_mask], vis=False)
                            if inlier_mask is None:
                                res[i_mask] = 0
                                labels_at_projected_pixels[i_mask] = 0
                            else:
                                outlier_id = np.where(i_mask)[0][~inlier_mask]
                                res[outlier_id] = 0          
                                labels_at_projected_pixels[outlier_id] = 0

                        res[res > 0] += point_instance_id.max()
                        point_instance_id[valid_mask] = res
                
                    ##########################
                    rows = np.nonzero(valid_mask)[0]
                    point_sem[rows[np.arange(labels_at_projected_pixels.shape[0])], labels_at_projected_pixels] += 1
                    ##########################

            world2cam_ls.append(torch.tensor(world2cam, device='cuda', dtype=torch.float32))

        ''' only keep points that are visible in at least one camera
        '''
        xyz_world = xyz_world[point_count > 0]
        point_sem = point_sem[point_count > 0]
        point_rgb_sum = point_rgb_sum[point_count > 0]
        xyz_lidar = xyz_lidar[point_count > 0]
        point_instance_id = point_instance_id[point_count > 0]
        point_count = point_count[point_count > 0]

        point_rgb = point_rgb_sum / point_count[:, None]
        
        # visualize_point_clouds(xyz_world[point_instance_id>0], point_instance_id[point_instance_id>0], 'preprocessed instance point cloud')
        # visualize_point_clouds(xyz_world, point_sem.argmax(1), 'preprocessed semantic point cloud')
        ####################################
        point_sem[:, 1] = 0
        point_sem[:, 9] = 0
        point_sem = point_sem+1e-6
        point_sem /= point_sem.sum(1, keepdims=True)

        point_cloud = {
            'point_xyz': torch.tensor(xyz_world.astype(np.float32), device='cuda'),
            'point_rgb': torch.tensor((point_rgb/255).astype(np.float32), device='cuda'),
            'point_sem': torch.tensor(point_sem.astype(np.float32), device='cuda'),
            'point_instance_id': torch.tensor(point_instance_id.astype(np.float32), device='cuda'),
            'intrinsic3x3_ls': intrinsic3x3_ls,
            'world2cam_ls': world2cam_ls,
        }
        train_cameras = [loadCam(id, c, 1, False, False).cuda() for id, c in enumerate(train_cam_infos)] 

    elif source == 'depth':
        train_cam_infos = []

        lidar_dir = join(path, "lidar")
        lidar2world = np.loadtxt(join(lidar_dir, f"{frame:0>2d}.txt"))
        lidar2world = old_world2new_world @ lidar2world
        world2lidar = np.linalg.inv(lidar2world)
        
        ego_dir = join(path, "ego")
        ego2world = np.loadtxt(join(ego_dir, f"{frame:0>2d}.txt"))
        ego2world = old_world2new_world @ ego2world
        world2ego = np.linalg.inv(ego2world)

        point_xyz_ls = []
        point_rgb_ls = []
        point_sem_ls = []
        world2cam_ls = []
        dynamic_mask_ls = []
        intrinsic3x3_scaled_ls = []

        # if not os.path.exists(os.path.join(path, f"vggt_0")):
        if True:
            for cam in range(6):
                save_dir = os.path.join(path, f"vggt_{cam}")
                os.makedirs(save_dir, exist_ok=True)
            global vggt_model, intrinsic3x3_scaled_vggt, prev_scale
            if vggt_model is None:
                print("Loading VGGT model... (Only the first time)")
                # This will automatically download the model weights the first time it's run, which may take a while.
                vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to('cuda')
                
                H_orig, W_orig = 900, 1600 
                H_target, W_target = 294, 518
                scale_x = W_target / W_orig
                scale_y = H_target / H_orig
                intrinsic3x3_scaled_vggt = intrinsic3x3_ls.copy()
                intrinsic3x3_scaled_vggt = np.stack(intrinsic3x3_scaled_vggt, axis=0)  # (6, 3, 3)
                intrinsic3x3_scaled_vggt[:, 0, 0] *= scale_x  # fx
                intrinsic3x3_scaled_vggt[:, 1, 1] *= scale_y  # fy
                intrinsic3x3_scaled_vggt[:, 0, 2] *= scale_x  # cx
                intrinsic3x3_scaled_vggt[:, 1, 2] *= scale_y  # cy
            prev_scale, depth_time, scale_time = process_frame_vggt(frame, path, intrinsic3x3_scaled_vggt, prev_scale, vggt_model)
            # print(f"Finished VGGT for {frame}, depth time: {depth_time*1000:.0f} ms, scale time: {scale_time*1000:.0f} ms")
        
        # if not os.path.exists(os.path.join(path, f"raft_0")):
        if True:
            for cam in range(6):
                save_dir = os.path.join(path, f"raft_{cam}")
                os.makedirs(save_dir, exist_ok=True)
            global raft_model, raft_padder, image_prev, depth_prev, cam2world_prev, intrinsic3x3_scaled_raft
            if raft_model is None:
                print("Loading RAFT model... (Only the first time)")
                raft_model, raft_padder = load_args("submodules/RAFT/raft-things.pth")
                intrinsic3x3_scaled_raft = torch.from_numpy(intrinsic3x3_scaled_vggt).to('cuda', dtype=torch.float32)
            start_time = time.time()
            if frame == 0:
                image_prev, depth_prev, cam2world_prev = process_frame_raft(frame, path, raft_model, raft_padder, intrinsic3x3_scaled_raft)
            else:
                image_prev, depth_prev, cam2world_prev = process_frame_raft(frame, path, raft_model, raft_padder, intrinsic3x3_scaled_raft, image_prev, depth_prev, cam2world_prev)
            # print(f"Finished RAFT for {frame}, time: {(time.time() - start_time)*1000:.0f} ms")
            
        for cam in range(6):
            depth_path = join(path, f"vggt_{cam}/{frame:0>2d}.npy")
            depth = np.load(depth_path)
            depth = cv2.resize(depth, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_NEAREST)
            height, width = depth.shape 

            name = f"{cam}/{frame:0>2d}"
            image_name = f"{name}.jpg"
            image_id = 0 #len(Image_ls)
            image_path = join(path, image_name)
            image = cv2.imread(image_path)
            org_width, org_height = image.shape[1], image.shape[0]
            image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (width, height), interpolation=cv2.INTER_LINEAR)

            cam2world = np.loadtxt(join(path, f"{name}.txt"))
            cam2world = old_world2new_world @ cam2world
            R_c2w = cam2world[:3, :3]
            T_c2w = cam2world[:3, 3]
            world2cam = np.linalg.inv(cam2world)
            R_w2c = world2cam[:3, :3]
            T_w2c = world2cam[:3, 3]
            world2cam_ls.append(torch.tensor(world2cam, device='cuda', dtype=torch.float32))
            
            semantic_path = join(path, f"{sem_prefix}_{cam}/{frame:0>2d}.png")
            if not os.path.exists(semantic_path):
                semantic_path = None
            fovx = focal2fov(intrinsic3x3_ls[cam][0,0], org_width)
            fovy = focal2fov(intrinsic3x3_ls[cam][1,1], org_height)
            cam_info = CameraInfo(uid=image_id, R=R_c2w, T=T_w2c, 
                FovY=fovy, FovX=fovx,
                depth_params=None,
                image_path=image_path, image_name=image_name, 
                depth_path="", 
                semantic_path=semantic_path,
                width=org_width, height=org_height, is_test=False)
            train_cam_infos.append(cam_info)
            scale_width = width / org_width
            scale_height = height / org_height 
            intrinsic3x3_scaled = intrinsic3x3_ls[cam].copy()
            intrinsic3x3_scaled[0, 0] *= scale_width
            intrinsic3x3_scaled[1, 1] *= scale_height
            intrinsic3x3_scaled[0, 2] *= scale_width
            intrinsic3x3_scaled[1, 2] *= scale_height
            intrinsic3x3_scaled_ls.append(intrinsic3x3_scaled)
            
            if frame > 0:
                dyn_mask_path = join(path, f"raft_{cam}/{frame:0>2d}_5mask.png")
                if not os.path.exists(dyn_mask_path):
                    dyn_mask = np.zeros((height, width), dtype=bool)
                else:
                    dyn_mask = cv2.imread(dyn_mask_path, -1)
                    dyn_mask = cv2.resize(dyn_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    dyn_mask = dyn_mask > 0
                dyn_mask = torch.tensor(dyn_mask, device='cuda')
                dynamic_mask_ls.append(dyn_mask)  # [N]
            if semantic_path is None:
                continue
            semantic_image = cv2.imread(semantic_path, -1)
            semantic_image = cv2.resize(semantic_image, (width, height), interpolation=cv2.INTER_NEAREST)
            valid_mask = semantic_image != 17  # shape: [H, W]
            if cam == 3: # back camera
                ego_mask = cv2.imread(f"ego_mask_{cam}.png", cv2.IMREAD_GRAYSCALE)
                ego_mask = cv2.resize(ego_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                valid_mask = valid_mask & (ego_mask > 0)  # remove ego car

            grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0)
            grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            threshold = np.percentile(grad_mag, 95)

            non_edge_mask = grad_mag <= threshold
            valid_mask = valid_mask & non_edge_mask

            xs, ys = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')  # shape: [W, H]
            xs = xs[valid_mask]  # [N]
            ys = ys[valid_mask]  # [N]
            depth_valid = depth[valid_mask]  # [N]

            
            point_xyz = unproject_to_world(
                xs, ys, depth_valid, intrinsic3x3_scaled, cam2world
            )  # [N, 3]

            semantic_image = semantic_image[valid_mask]
            image = image[valid_mask]  # [N, 3]
            point_xyz_ls.append(point_xyz)  # [N, 3]
            point_rgb_ls.append(image)  # [N, 3]
            point_sem_ls.append(semantic_image)  # [N]
            
            
        point_xyz = np.concatenate(point_xyz_ls, axis=0)
        point_rgb = np.concatenate(point_rgb_ls, axis=0)
        point_sem = np.concatenate(point_sem_ls, axis=0)
        point_sem = np.eye(num_classes)[point_sem]
        point_xyz, point_rgb, point_sem = voxelize(point_xyz, point_rgb, point_sem, voxel_size=0.05)
        remain_mask = remove_radius_outlier_numpy(point_xyz, 0.2, 2)
        # print('remain ratio:', np.sum(remain_mask) / len(remain_mask))
        point_xyz = point_xyz[remain_mask]
        point_rgb = point_rgb[remain_mask]
        point_sem = point_sem[remain_mask]
        

        point_xyz = torch.from_numpy(point_xyz).float().cuda()
        point_rgb = torch.from_numpy(point_rgb).float().cuda()
        point_sem = torch.from_numpy(point_sem).float().cuda()
        point_rgb = point_rgb / 255.0
        point_sem[:, 1] = 0
        point_sem[:, 9] = 0
        point_sem = point_sem+1e-6
        point_sem /= point_sem.sum(1, keepdims=True)
        point_cloud = {
            'point_xyz': point_xyz,
            'point_rgb': point_rgb,
            'point_sem': point_sem,
            'world2cam_ls': world2cam_ls,
            'dynamic_mask_ls': dynamic_mask_ls,
            'intrinsic3x3_ls': intrinsic3x3_scaled_ls,
        }
        train_cameras = [loadCam(id, c, 2, False, False).cuda() for id, c in enumerate(train_cam_infos)]
    # for k, v in point_cloud.items():
    #     print(k, v.shape)
    return {
        "train_data": point_cloud,
        "train_cameras": train_cameras,
        "world2ego": world2ego,
        "world2lidar": world2lidar,
    }


