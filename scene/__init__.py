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
from scene.gaussian_model_A_3DGS import GaussianModel as GaussianModel_A
from scene.gaussian_model_B_scale_aware_voxel import GaussianModel as GaussianModel_B
from scene.gaussian_model_C_add_hist import GaussianModel as GaussianModel_C
from scene.gaussian_model_D_track_hist import GaussianModel as GaussianModel_D
from scene.gaussian_model_E_fusion import GaussianModel as GaussianModel_E

from arguments import settings, semantic_list
import torch
import ipdb
import time
from threading import Thread
from concurrent.futures import Future
from custom_utils import *

class Scene:

    @torch.no_grad()
    def async_load(self, frame_id):
        self.next_load = Future()
        try:
            scene_info = load_nuscenes(self.scene_path, frame_id, self.old_world2new_world, self.intrinsic3x3_ls, self.opt.variant)
            point_cloud = scene_info['train_data']
            if self.opt.occ_setting == "Occ3D":
                world2local_np = scene_info['world2ego']
            elif self.opt.occ_setting == "nuCraft":
                world2local_np = scene_info['world2lidar']
            else:
                raise ValueError("Invalid occ_setting")
            world2local_ts = torch.tensor(world2local_np, device='cuda', dtype=torch.float32)

            train_cameras = scene_info['train_cameras']
            train_cameras_this_timestep = train_cameras*self.opt.iterations

            self.next_load.set_result((train_cameras_this_timestep, point_cloud, world2local_np, world2local_ts))
        except Exception as e:
            self.next_load.set_exception(e)
    
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
        self.load_intrinsics()
        if self.opt.multi_thread:
            Thread(target=self.async_load, args=(0, )).start()
        else:
            self.async_load(0)
        self.train_cameras_this_timestep = []
        self.total_frame = int((len(os.listdir(join(self.scene_path, "0")))-1)/2)
        root, self.scene = os.path.split(self.scene_path)
        self.model_path = model_params.model_path
        if self.opt.save_ply:
            self.ply_path = os.path.join(self.model_path, "point_cloud")
            os.makedirs(self.ply_path, exist_ok=True)
        if self.opt.save_occ:
            self.occ_path = os.path.join(self.model_path, "Occ")
            os.makedirs(self.occ_path, exist_ok=True)
        if self.opt.eval_occ:
            self.mapping = json.load(open(os.path.join(root, "mapping.json")))[self.scene]
            self.hist_occ3d_lidar = 0
            self.hist_occ3d_camera = 0
            self.hist_nucraft = 0
            self.ious, self.mious = [], []
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
        if frame_id < self.total_frame - 1:
            if self.opt.multi_thread:
                Thread(target=self.async_load, args=(frame_id+1, )).start()
            else:
                self.async_load(frame_id+1)
        self.train_cameras_this_timestep = train_cameras_this_timestep
        random.shuffle(self.train_cameras_this_timestep)

        self.xyz = point_cloud['point_xyz']
        self.update_time += time.time() - start
        if self.gaussians.N == 0:
            self.gaussians.create_from_pcd(point_cloud)
        else:
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
                Thread(target=self.gaussians.save_ply, args=(point_cloud_dir, )).start()
            else:
                self.gaussians.save_ply(point_cloud_dir)
        if self.opt.save_occ:
            start_time = time.time()
            voxel_indices_from_gs, voxel_cls_from_gs = self.gaussians.gs2voxel(self.world2local_ts, *settings[self.opt.occ_setting], self.gaussians.detach_dict)
            self.voxelization_time += time.time() - start_time
            voxel_cls_from_gs[voxel_cls_from_gs == 1] = 0
            voxel_cls_from_gs[voxel_cls_from_gs == 9] = 0
            # if self.opt.eval_occ_each_time:
            #     self.eval(timestep, voxel_indices_from_gs, voxel_cls_from_gs)
            torch.save({"voxel_indices": voxel_indices_from_gs, "voxel_cls": voxel_cls_from_gs}, join(self.occ_path, f"{timestep:02d}.pth"))
            
            
    def eval(self, timestep, voxel_indices_from_gs, voxel_cls_from_gs):
        if self.opt.occ_setting == "Occ3D":
            new_hist_occ_lidar, new_hist_occ_camera, dense_cls_occ_np = eval_occ(timestep, voxel_indices_from_gs, voxel_cls_from_gs, "Occ3D", self.scene, self.mapping, occ3d_path=self.opt.occ3d_path)
            self.hist_occ3d_lidar += new_hist_occ_lidar
            self.hist_occ3d_camera += new_hist_occ_camera
            iou, miou, mious = cal_iou_miou(new_hist_occ_camera)
            print(f"Occ3D: timestep: {timestep}, IoU: {iou:.4f}, mIoU: {miou:.4f}")
            # if self.vis_occ is not None:
                # voxel_indices_occ, voxel_color_ids_occ = dense2sparse(dense_cls_occ_np)
                # view_sparse_voxel([(voxel_indices_occ, voxel_color_ids_occ), (voxel_indices_from_gs.cpu().numpy(), voxel_cls_from_gs.cpu().numpy())], self.vis_occ)
        elif self.opt.occ_setting == "nuCraft":
            new_hist_nu, nucraft_gt = eval_occ(timestep, voxel_indices_from_gs, voxel_cls_from_gs, "nuCraft", self.scene, self.mapping, nucraft_path=self.opt.nucraft_path)
            self.hist_nucraft += new_hist_nu
            iou, miou, mious = cal_iou_miou(new_hist_nu)
            print(f"nuCraft: timestep: {timestep}, IoU: {iou:.4f}, mIoU: {miou:.4f}")

            # if self.vis_occ is not None:
            #     view_sparse_voxel([nucraft_gt, (voxel_indices_from_gs.cpu().numpy(), voxel_cls_from_gs.cpu().numpy())], self.vis_occ)

        # mask = voxel_cls_from_gs != 0
        # voxel_indices_from_gs = voxel_indices_from_gs[mask]
        # voxel_cls_from_gs = voxel_cls_from_gs[mask]
        # view_sparse_voxel([(voxel_indices_from_gs.cpu().numpy(), voxel_cls_from_gs.cpu().numpy())], self.vis_occ, f"vis/gradually/{self.scene}_{timestep}.png")


    def getTrainCameras(self, iteration):
        return self.train_cameras_this_timestep[iteration]
    
    @property
    def len_train_cameras(self):
        return len(self.train_cameras_this_timestep)


    def report_results(self):
        self.result = {
            "update_time": f"{self.update_time/self.total_frame*1000:.0f}ms", 
            "training_time": f"{self.training_time/self.total_frame*1000:.0f}ms", 
            "fusion_time": f"{self.fusion_time/self.total_frame*1000:.0f}ms", 
            "voxelization_time": f"{self.voxelization_time/self.total_frame*1000:.0f}ms",
        }

        if self.opt.eval_occ and self.opt.save_occ:
            Occ_list = sorted([file for file in os.listdir(join(self.model_path, "Occ"))])
            for file in Occ_list:
                timestep = int(file.split('.')[0])
                voxel_indices, voxel_cls = torch.load(join(self.model_path, "Occ", file)).values()
                self.eval(timestep, voxel_indices, voxel_cls)

        
            if not isinstance(self.hist_occ3d_camera, int):
                self.result.update({
                    'Occ3D_cameramask': {
                        'Avg': {},
                        # 'Per_frame': {}
                    },
                })
                self.result['Occ3D_cameramask']['Avg']['iou'], self.result['Occ3D_cameramask']['Avg']['miou'], self.result['Occ3D_cameramask']['Avg']['mious'] = cal_iou_miou(self.hist_occ3d_camera)
                self.result['Occ3D_cameramask']['Avg']['mious'] = {f'{semantic_list[i]}': float(self.result['Occ3D_cameramask']['Avg']['mious'][i]) for i in range(len(self.result['Occ3D_cameramask']['Avg']['mious']))}
                print("Occ3D_cameramask: ", self.result['Occ3D_cameramask']['Avg']['iou'], self.result['Occ3D_cameramask']['Avg']['miou'])
            
            if not isinstance(self.hist_occ3d_lidar, int):    
                self.result.update({
                    'Occ3D_lidarmask': {
                        'Avg': {},
                        # 'Per_frame': {}
                    },
                })
                self.result['Occ3D_lidarmask']['Avg']['iou'], self.result['Occ3D_lidarmask']['Avg']['miou'], self.result['Occ3D_lidarmask']['Avg']['mious'] = cal_iou_miou(self.hist_occ3d_lidar)
                self.result['Occ3D_lidarmask']['Avg']['mious'] = {f'{semantic_list[i]}': float(self.result['Occ3D_lidarmask']['Avg']['mious'][i]) for i in range(len(self.result['Occ3D_lidarmask']['Avg']['mious']))}
                print("Occ3D_lidarmask: ", self.result['Occ3D_lidarmask']['Avg']['iou'], self.result['Occ3D_lidarmask']['Avg']['miou'])

            if not isinstance(self.hist_nucraft, int):
                self.result.update({
                    'nuCraft': {
                        'Avg': {},
                        # 'Per_frame': {}
                    },
                })
                self.result['nuCraft']['Avg']['iou'], self.result['nuCraft']['Avg']['miou'], self.result['nuCraft']['Avg']['mious'] = cal_iou_miou(self.hist_nucraft)
                self.result['nuCraft']['Avg']['mious'] = {f'{semantic_list[i]}': float(self.result['nuCraft']['Avg']['mious'][i]) for i in range(len(self.result['nuCraft']['Avg']['mious']))}
                print("nuCraft: ", self.result['nuCraft']['Avg']['iou'], self.result['nuCraft']['Avg']['miou'])

            
            file_path = join(self.model_path, "result.json")
            json_data = json.dumps(self.result, default=custom_encoder, indent=4)
            with open(file_path, "w") as f:
                f.write(json_data)
        for key, value in self.result.items():
            if key.endswith('time'):
                print(f"{key}: {value}")
        

        

