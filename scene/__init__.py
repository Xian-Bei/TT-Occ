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
from utils.testtime_precompute import ScenePrecomputeManager


def get_autocast_context():
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    return torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype)


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
            self.precompute_manager.wait_prefetch_frame(self.scene_path, frame_id)
            self.precompute_manager.ensure_frame_features(self.scene_path, frame_id, self.opt.variant, self.opt)
        memory_features = self.precompute_manager.get_cached_frame_features(self.scene_path, frame_id)
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
            memory_features=memory_features,
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
            self.precompute_manager.prefetch_frame(self.scene_path, frame_id, self.opt.variant, self.opt)
    
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
        if self.opt.eval_occ:
            self.mapping = json.load(open(os.path.join(root, "mapping.json")))[self.scene]
        self.hist_occ3d_camera = torch.zeros((num_classes + 1, num_classes + 1), device='cuda')
        self.hist_nucraft = torch.zeros((num_classes + 1, num_classes + 1), device='cuda')
        # self.vis_occ = VoxelGridVisualizer('Comparison', 1, 600) if self.opt.vis_occ else None
        # os.makedirs(join(self.model_path, "Occ"), exist_ok=True)
        self.update_time = 0
        self.precompute_total_time = 0
        self.precompute_semantic_time = 0
        self.precompute_depth_time = 0
        self.precompute_raft_time = 0
        self.precompute_raft_warmup_time = 0
        self.precompute_disk_hit = 0
        self.precompute_memory_hit = 0
        self.precompute_computed = 0
        self.voxelization_time = 0
        self.fusion_time = 0
        self.training_time = 0
        self.scene_start = time.time()

    @torch.no_grad()
    def update_scene(self, frame_id):
        start = time.time()
        train_cameras_this_timestep, point_cloud, \
            self.world2local_np, self.world2local_ts = self.next_load.result()
        pre_profile = self.precompute_manager.get_frame_profile(self.scene_path, frame_id)
        self.precompute_total_time += pre_profile.get("total_s", 0.0)
        self.precompute_semantic_time += pre_profile.get("semantic_s", 0.0)
        self.precompute_depth_time += pre_profile.get("depth_s", 0.0)
        self.precompute_raft_time += pre_profile.get("raft_s", 0.0)
        self.precompute_raft_warmup_time += pre_profile.get("raft_warmup_s", 0.0)
        status = pre_profile.get("status", "unknown")
        if status == "disk_hit":
            self.precompute_disk_hit += 1
        elif status == "memory_hit":
            self.precompute_memory_hit += 1
        elif status == "computed":
            self.precompute_computed += 1
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
        with get_autocast_context():
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
            with get_autocast_context():
                voxel_indices_from_gs, voxel_cls_from_gs = self.gaussians.gs2voxel(
                    self.world2local_ts,
                    *settings[self.opt.occ_setting],
                    self.gaussians.detach_dict,
                )
            self.voxelization_time += time.time() - start_time
            voxel_cls_from_gs[voxel_cls_from_gs == 1] = 0
            voxel_cls_from_gs[voxel_cls_from_gs == 9] = 0
            if self.opt.eval_online and self.opt.eval_occ:
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
            "precompute_total_time": f"{self.precompute_total_time/self.total_frame*1000:.0f}ms",
            "precompute_semantic_time": f"{self.precompute_semantic_time/self.total_frame*1000:.0f}ms",
            "precompute_depth_time": f"{self.precompute_depth_time/self.total_frame*1000:.0f}ms",
            "precompute_raft_time": f"{self.precompute_raft_time/self.total_frame*1000:.0f}ms",
            "precompute_raft_warmup_time": f"{self.precompute_raft_warmup_time/self.total_frame*1000:.0f}ms",
            "precompute_frames_computed": self.precompute_computed,
            "precompute_frames_disk_hit": self.precompute_disk_hit,
            "precompute_frames_memory_hit": self.precompute_memory_hit,
        }

        if self.opt.eval_online and self.opt.eval_occ:
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
                if self.opt.verbose:
                    print(f"{result_key}: iou={iou:.4f}, miou={miou:.4f}")

            file_path = join(self.model_path, "result.json")
            json_data = json.dumps(self.result, default=custom_encoder, indent=4)
            with open(file_path, "w") as f:
                f.write(json_data)
        if self.opt.verbose:
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        

