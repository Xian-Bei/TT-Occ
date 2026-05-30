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
import random
import json
from scene.dataset_readers import *
import itertools
from scene.gaussian_model import GaussianModel

from arguments import settings, semantic_list
import torch
import time
from concurrent.futures import Future, ThreadPoolExecutor
from custom_utils import *
from utils.ray_metrics_cuda import generate_lidar_rays, main_rayiou
from utils.occ_eval import get_lidar_origin_in_lidar_ref, lidar_origin_in_ego
from utils.testtime_precompute import ScenePrecomputeManager

class Scene:

    def _to_cuda(self, data):
        if isinstance(data, torch.Tensor):
            return data.cuda(non_blocking=True)
        return data

    def _to_cuda_recursive(self, data):
        if isinstance(data, dict):
            return {k: self._to_cuda_recursive(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._to_cuda_recursive(v) for v in data]
        if isinstance(data, tuple):
            return tuple(self._to_cuda_recursive(v) for v in data)
        return self._to_cuda(data)

    @torch.no_grad()
    def load_frame(self, frame_id):
        if self.opt.enable_testtime_precompute:
            self.precompute_manager.wait_prefetch(self.scene_path)
            self.precompute_manager.ensure_scene_features(self.scene_path, self.opt.variant, self.opt)
        scene_info = load_nuscenes(
            self.scene_path,
            frame_id,
            self.old_world2new_world,
            self.intrinsic3x3_ls,
            source=self.opt.variant,
            semantic_prefix=self.opt.semantic_prefix,
            depth_prefix=self.opt.depth_prefix,
            depth_ext=self.opt.depth_ext,
            dynamic_mask_prefix=self.opt.dynamic_mask_prefix,
            dynamic_mask_suffix=self.opt.dynamic_mask_suffix,
            feature_root=self.opt.feature_root,
        )
        point_cloud = scene_info['train_data']
        if self.opt.occ_setting == "Occ3D":
            world2local_np = scene_info['world2ego']
        elif self.opt.occ_setting == "nuCraft":
            world2local_np = scene_info['world2lidar']
        else:
            raise ValueError("Invalid occ_setting")
        world2local_ts = torch.tensor(world2local_np, dtype=torch.float32)

        train_cameras = scene_info['train_cameras']
        train_cameras_this_timestep = train_cameras*self.opt.iterations
        return train_cameras_this_timestep, point_cloud, world2local_np, world2local_ts

    def schedule_load(self, frame_id):
        if self.opt.multi_thread:
            self.next_load = self.load_executor.submit(self.load_frame, frame_id)
        else:
            future = Future()
            try:
                future.set_result(self.load_frame(frame_id))
            except Exception as e:
                future.set_exception(e)
            self.next_load = future
        if self.opt.enable_testtime_precompute and self.opt.precompute_prefetch:
            self.precompute_manager.prefetch_scene(self.scene_path, self.opt.variant, self.opt)
    
    def load_intrinsics(self):
        new_world2old_world = np.loadtxt(join(self.scene_path, "0/00.txt"))
        self.old_world2new_world = np.linalg.inv(new_world2old_world)
        self.intrinsic3x3_ls = []
        for cam in range(6):
            intrinsic3x3 = np.loadtxt(join(self.scene_path, f"{cam}/intrinsic.txt"))
            self.intrinsic3x3_ls.append(intrinsic3x3)

        

    def __init__(self, model_params, optimization_params, pipeline_params, gaussians, background):
        self.scene_path = model_params.source_path
        self.pipe = pipeline_params
        self.background = background
        self.opt = optimization_params
        self.gaussians = gaussians
        self.precompute_manager = ScenePrecomputeManager.get()
        self.load_intrinsics()
        self.load_executor = ThreadPoolExecutor(max_workers=1) if self.opt.multi_thread else None
        self.save_executor = ThreadPoolExecutor(max_workers=1) if self.opt.multi_thread else None
        self.save_futures = []
        self.schedule_load(0)
        self.train_cameras_this_timestep = []
        self.total_frame = int((len(os.listdir(join(self.scene_path, "0")))-1)/2)
        root, self.scene = os.path.split(self.scene_path)
        self.data_root = root
        self.model_path = model_params.model_path
        self.model_name = os.path.basename(os.path.dirname(self.model_path))
        if self.opt.save_ply:
            self.ply_path = os.path.join(self.model_path, "point_cloud")
            os.makedirs(self.ply_path, exist_ok=True)
        if self.opt.save_occ:
            self.occ_path = os.path.join(self.model_path, "Occ")
            os.makedirs(self.occ_path, exist_ok=True)
        if self.opt.eval_occ or self.opt.eval_rayiou:
            self.mapping = json.load(open(os.path.join(root, "mapping.json")))[self.scene]
        self.hist_occ3d_camera = torch.zeros((num_classes + 1, num_classes + 1), device='cuda')
        self.hist_nucraft = torch.zeros((num_classes + 1, num_classes + 1), device='cuda')
        self.rayiou_occ3d_pred = []
        self.rayiou_occ3d_gt = []
        self.rayiou_occ3d_origin = []
        self.rayiou_nucraft_pred = []
        self.rayiou_nucraft_gt = []
        self.lidar_rays = torch.from_numpy(generate_lidar_rays()).to(device='cuda', dtype=torch.float32)
        # self.vis_occ = VoxelGridVisualizer('Comparison', 1, 600) if self.opt.vis_occ else None
        # os.makedirs(join(self.model_path, "Occ"), exist_ok=True)
        self.update_time = 0
        self.voxelization_time = 0
        self.fusion_time = 0
        self.training_time = 0
        self.scene_start = time.time()

    @torch.no_grad()
    def update_scene(self, frame_id):
        start = time.time()
        train_cameras_this_timestep, point_cloud, \
            self.world2local_np, self.world2local_ts = self.next_load.result()
        self.train_cameras_this_timestep = [cam.cuda() for cam in train_cameras_this_timestep]
        point_cloud = self._to_cuda_recursive(point_cloud)
        self.world2local_ts = self.world2local_ts.cuda(non_blocking=True)
        if frame_id < self.total_frame - 1:
            self.schedule_load(frame_id+1)
        random.shuffle(self.train_cameras_this_timestep)

        self.xyz = point_cloud['point_xyz']
        self.update_time += time.time() - start
        if self.gaussians.N == 0:
            self.gaussians.create_from_pcd(point_cloud)
        else:
            self.gaussians.update_detach_for_async_save_eval()
            self.gaussians.add_from_pcd(point_cloud)
        
            
            
            
            
            
    @torch.no_grad()
    def save_eval(self, timestep):
        self.gaussians.update_detach_for_async_save_eval()
        start_time = time.time()    
        self.gaussians.fusion(K=self.opt.K)
        self.fusion_time += time.time() - start_time
        if self.opt.save_ply:
            point_cloud_dir = os.path.join(self.ply_path, f"iteration_{timestep:02d}")
            if self.opt.multi_thread:
                self.save_futures.append(self.save_executor.submit(self.gaussians.save_ply, point_cloud_dir))
            else:
                self.gaussians.save_ply(point_cloud_dir)
        if self.opt.save_occ:
            start_time = time.time()
            voxel_indices_from_gs, voxel_cls_from_gs = self.gaussians.gs2voxel(self.world2local_ts, *settings[self.opt.occ_setting], self.gaussians.detach_dict)
            self.voxelization_time += time.time() - start_time
            voxel_cls_from_gs[voxel_cls_from_gs == 1] = 0
            voxel_cls_from_gs[voxel_cls_from_gs == 9] = 0
            if self.opt.eval_online and (self.opt.eval_occ or self.opt.eval_rayiou):
                self.eval(timestep, voxel_indices_from_gs, voxel_cls_from_gs)
            occ_path = join(self.occ_path, f"{timestep:02d}.pth")
            occ_payload = {
                "voxel_indices": voxel_indices_from_gs.cpu(),
                "voxel_cls": voxel_cls_from_gs.cpu(),
            }
            if self.opt.multi_thread:
                self.save_futures.append(self.save_executor.submit(torch.save, occ_payload, occ_path))
            else:
                torch.save(occ_payload, occ_path)
            
            
    def eval(self, timestep, voxel_indices_from_gs, voxel_cls_from_gs):
        if self.opt.occ_setting == "Occ3D":
            if self.opt.eval_occ:
                new_hist_occ_camera, _ = eval_occ(
                    timestep,
                    voxel_indices_from_gs,
                    voxel_cls_from_gs,
                    "Occ3D",
                    self.scene,
                    self.mapping,
                    gt_path=self.opt.occ3d_path,
                )
                self.hist_occ3d_camera += new_hist_occ_camera
                iou, miou, _ = cal_iou_miou(new_hist_occ_camera)
                print(f"Occ3D: timestep: {timestep}, IoU: {iou:.4f}, mIoU: {miou:.4f}")

            if self.opt.eval_rayiou and self.opt.occ3d_path:
                dense_cls_occ_np, mask = load_occ3d_gt(
                    join(self.opt.occ3d_path, self.scene, f"{self.mapping['sample_token'][f'{timestep:0>2d}']}/labels.npz")
                )
                dense_cls_occ = torch.tensor(dense_cls_occ_np, device='cuda', dtype=torch.long)
                dense_cls_from_gs = sparse2dense_torch(
                    voxel_indices_from_gs.long(),
                    voxel_cls_from_gs.long(),
                    *settings['Occ3D'],
                )
                mask_ts = torch.tensor(mask, device='cuda')
                dense_cls_occ[~mask_ts] = 17
                dense_cls_from_gs[~mask_ts] = 17

                tp_ls = []
                ego2world = np.loadtxt(join(self.data_root, self.scene, 'ego', f'{timestep:02d}.txt'))
                for i in range(0, self.total_frame, 5):
                    lidar2world = np.loadtxt(join(self.data_root, self.scene, 'lidar', f'{i:02d}.txt'))
                    lidar_origin_ego = lidar_origin_in_ego(ego2world, lidar2world).reshape(1, 1, 3)
                    tp_ls.append(torch.tensor(lidar_origin_ego, device='cpu'))
                lidar_origin_ego = torch.cat(tp_ls, dim=1)
                self.rayiou_occ3d_origin.append(lidar_origin_ego)
                self.rayiou_occ3d_gt.append(dense_cls_occ.cpu().numpy().astype(np.uint8))
                self.rayiou_occ3d_pred.append(dense_cls_from_gs.cpu().numpy().astype(np.uint8))

        elif self.opt.occ_setting == "nuCraft":
            if self.opt.eval_occ:
                new_hist_nu, _ = eval_occ(
                    timestep,
                    voxel_indices_from_gs,
                    voxel_cls_from_gs,
                    "nuCraft",
                    self.scene,
                    self.mapping,
                    gt_path=self.opt.nucraft_path,
                )
                self.hist_nucraft += new_hist_nu
                iou, miou, _ = cal_iou_miou(new_hist_nu)
                print(f"nuCraft: timestep: {timestep}, IoU: {iou:.4f}, mIoU: {miou:.4f}")

            if self.opt.eval_rayiou and self.opt.nucraft_path:
                lidar2world_ref = np.loadtxt(join(self.data_root, self.scene, 'lidar', f'{timestep:02d}.txt'))
                tp_ls = []
                for i in range(0, self.total_frame, 5):
                    lidar2world = np.loadtxt(join(self.data_root, self.scene, 'lidar', f'{i:02d}.txt'))
                    lidar_origin_lidar_ref = get_lidar_origin_in_lidar_ref(
                        lidar2world_ref, lidar2world
                    ).reshape(1, 1, 3)
                    tp_ls.append(torch.tensor(lidar_origin_lidar_ref, device='cpu'))
                lidar_origins = torch.cat(tp_ls, dim=1).to('cuda')
                nucraft_gt = load_nucraft_gt(join(self.opt.nucraft_path, f"{self.mapping['LIDAR_TOP'][f'{timestep:0>2d}']}.bin"))
                dense_cls_gt = torch.tensor(
                    sparse2dense(*nucraft_gt, *settings['nuCraft_np']).astype(np.uint8),
                    device='cuda',
                    dtype=torch.long,
                )
                dense_cls_pred = sparse2dense_torch(
                    voxel_indices_from_gs.long(),
                    voxel_cls_from_gs.long(),
                    *settings['nuCraft'],
                )
                pcd_pred = process_one_sample(
                    sem_pred=dense_cls_pred,
                    lidar_rays=self.lidar_rays,
                    output_origin=lidar_origins,
                    occ_class_names=semantic_list,
                )
                pcd_gt = process_one_sample(
                    sem_pred=dense_cls_gt,
                    lidar_rays=self.lidar_rays,
                    output_origin=lidar_origins,
                    occ_class_names=semantic_list,
                )
                free_id = len(semantic_list) - 1
                valid_mask = (pcd_gt[:, 0].long() != free_id)
                self.rayiou_nucraft_pred.append(pcd_pred[valid_mask])
                self.rayiou_nucraft_gt.append(pcd_gt[valid_mask])


    def getTrainCameras(self, iteration):
        return self.train_cameras_this_timestep[iteration]
    
    @property
    def len_train_cameras(self):
        return len(self.train_cameras_this_timestep)


    def report_results(self):
        self.wait_for_saves()
        self.result = {
            "update_time": f"{self.update_time/self.total_frame*1000:.0f}ms", 
            "training_time": f"{self.training_time/self.total_frame*1000:.0f}ms", 
            "fusion_time": f"{self.fusion_time/self.total_frame*1000:.0f}ms", 
            "voxelization_time": f"{self.voxelization_time/self.total_frame*1000:.0f}ms",
        }

        if self.opt.eval_online and (self.opt.eval_occ or self.opt.eval_rayiou):
            if self.opt.eval_occ:
                if self.opt.occ_setting == "Occ3D":
                    iou, miou, mious = cal_iou_miou(self.hist_occ3d_camera)
                    result_key = 'Occ3D_cameramask'
                elif self.opt.occ_setting == "nuCraft":
                    iou, miou, mious = cal_iou_miou(self.hist_nucraft)
                    result_key = 'nuCraft'
                else:
                    raise ValueError(f"Unsupported occ_setting: {self.opt.occ_setting}")
                mious_dict = {f'{semantic_list[i]}': float(mious[i]) for i in range(len(mious))}
                self.result[result_key] = {
                    'Avg': {
                        'iou': float(iou),
                        'miou': float(miou),
                        'mious': mious_dict,
                    },
                }
                print(f"{result_key}: iou={iou:.4f}, miou={miou:.4f}")

            if self.opt.eval_rayiou:
                if self.opt.occ_setting == "Occ3D" and len(self.rayiou_occ3d_pred) > 0:
                    print(f"RayIoU Occ3D ({self.scene}):")
                    occ3d_ray = main_rayiou(
                        self.rayiou_occ3d_pred,
                        self.rayiou_occ3d_gt,
                        self.rayiou_occ3d_origin,
                        occ_class_names=semantic_list,
                    )
                    self.result['Occ3D_rayiou'] = {k: v for k, v in occ3d_ray.items() if k != 'table'}
                if self.opt.occ_setting == "nuCraft" and len(self.rayiou_nucraft_pred) > 0:
                    print(f"RayIoU nuCraft ({self.scene}):")
                    iou_list = calc_rayiou(self.rayiou_nucraft_pred, self.rayiou_nucraft_gt, semantic_list)
                    rayiou_0 = torch.nanmean(iou_list[0])
                    rayiou_1 = torch.nanmean(iou_list[1])
                    rayiou_2 = torch.nanmean(iou_list[2])
                    rayiou = torch.nanmean(torch.stack([iou_list[0], iou_list[1], iou_list[2]], dim=0))
                    self.result['nuCraft_rayiou'] = {
                        'RayIoU': rayiou.item(),
                        'RayIoU@1': rayiou_0.item(),
                        'RayIoU@2': rayiou_1.item(),
                        'RayIoU@4': rayiou_2.item(),
                    }

            file_path = join(self.model_path, "result.json")
            json_data = json.dumps(self.result, default=custom_encoder, indent=4)
            with open(file_path, "w") as f:
                f.write(json_data)
        for key, value in self.result.items():
            if key.endswith('time'):
                print(f"{key}: {value}")
        
    def wait_for_saves(self):
        for future in self.save_futures:
            future.result()
        self.save_futures.clear()

    def close(self):
        self.wait_for_saves()
        if self.load_executor is not None:
            self.load_executor.shutdown(wait=True)
        if self.save_executor is not None:
            self.save_executor.shutdown(wait=True)
        if hasattr(self, "precompute_manager"):
            self.precompute_manager.close()

        

