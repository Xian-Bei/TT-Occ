"""Shared occupancy evaluation (mIoU) used by train and summary scripts."""

import os
from os.path import join

import numpy as np
import torch
from arguments import semantic_list, settings
from custom_utils import (
    cal_iou_miou,
    eval_occ,
    eval_occ_2,
    load_nucraft_gt,
    load_occ3d_gt,
    sparse2dense,
    sparse2dense_torch,
    transform_occ_from_ego_to_lidar,
    transform_occ_from_ego_to_lidar_torch,
)


def eval_miou_hist_for_scene(scene_dir, scene, setting, mapping_scene, gt_path, data_path=None, model_name=""):
    hist_one_scene = 0
    occ_ls = sorted([f for f in os.listdir(join(scene_dir, 'Occ'))])
    if len(occ_ls) < 38:
        raise ValueError(
            f"Not enough occupancy files in {scene_dir}/Occ. Found {len(occ_ls)}, expected at least 38."
        )
    for occ_file in occ_ls:
        ckpt = torch.load(join(scene_dir, 'Occ', occ_file), map_location='cpu')
        timestep = int(occ_file.split('_')[-1].split('.')[0])
        if model_name.startswith('selfocc'):
            if data_path is None:
                data_path = os.path.dirname(os.path.dirname(scene_dir))
            ego2world = np.loadtxt(join(data_path, scene, 'ego', f'{timestep:02d}.txt'))
            lidar2world_ref = np.loadtxt(join(data_path, scene, 'lidar', f'{timestep:02d}.txt'))
            ego2lidar = np.matmul(np.linalg.inv(lidar2world_ref), ego2world)
            dense_cls_from_gs = transform_occ_from_ego_to_lidar(
                ckpt['voxel_indices'].cpu().numpy(),
                ckpt['voxel_cls'].cpu().numpy(),
                ego2lidar,
            )
            dense_cls_from_gs = torch.tensor(dense_cls_from_gs, device='cuda')
            cnm_mask = dense_cls_from_gs == 255
            dense_cls_from_gs[cnm_mask] = 17
            hist_one_frame, _ = eval_occ_2(
                timestep, dense_cls_from_gs, setting, scene, mapping_scene, gt_path=gt_path, cnm_mask=cnm_mask
            )
        else:
            hist_one_frame, _ = eval_occ(
                timestep,
                ckpt['voxel_indices'],
                ckpt['voxel_cls'],
                setting,
                scene,
                mapping_scene,
                gt_path=gt_path,
            )
        hist_one_scene += hist_one_frame
    return hist_one_scene


def eval_miou_for_scene(scene_dir, scene, setting, mapping_scene, gt_path, data_path=None, model_name=""):
    hist = eval_miou_hist_for_scene(
        scene_dir, scene, setting, mapping_scene, gt_path, data_path=data_path, model_name=model_name
    )
    return cal_iou_miou(hist)
