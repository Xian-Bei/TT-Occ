"""Shared occupancy evaluation (mIoU / RayIoU) used by train and summary scripts."""

import os
from os.path import join

import numpy as np
import torch
from prettytable import PrettyTable

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
from utils.ray_metrics_cuda import calc_rayiou, generate_lidar_rays, main_rayiou, process_one_sample

OCC_CLASS_NAMES = semantic_list


def get_lidar_origin_in_lidar_ref(T_lidar2world_ref, T_lidar2world):
    lidar_origin_world = T_lidar2world[:3, 3]
    lidar_origin_world_h = np.concatenate([lidar_origin_world, [1]])
    T_world2lidar_ref = np.linalg.inv(T_lidar2world_ref)
    lidar_origin_lidar_ref = T_world2lidar_ref @ lidar_origin_world_h
    return lidar_origin_lidar_ref[:3]


def lidar_origin_in_ego(T_ego2world, T_lidar2world):
    lidar_origin_world = T_lidar2world[:3, 3]
    lidar_origin_world_h = np.concatenate([lidar_origin_world, [1]])
    T_world2ego = np.linalg.inv(T_ego2world)
    lidar_origin_ego = T_world2ego @ lidar_origin_world_h
    return lidar_origin_ego[:3]


def _format_rayiou_result(iou_list, occ_class_names=OCC_CLASS_NAMES):
    rayiou_0 = torch.nanmean(iou_list[0])
    rayiou_1 = torch.nanmean(iou_list[1])
    rayiou_2 = torch.nanmean(iou_list[2])
    rayiou = torch.nanmean(torch.stack([iou_list[0], iou_list[1], iou_list[2]], dim=0))

    table = PrettyTable(['Class Names', 'RayIoU@1', 'RayIoU@2', 'RayIoU@4'])
    table.float_format = '.3'
    for i in range(len(occ_class_names) - 1):
        r1 = iou_list[0][i].item() if torch.isfinite(iou_list[0][i]) else float('nan')
        r2 = iou_list[1][i].item() if torch.isfinite(iou_list[1][i]) else float('nan')
        r4 = iou_list[2][i].item() if torch.isfinite(iou_list[2][i]) else float('nan')
        table.add_row([occ_class_names[i], r1, r2, r4], divider=(i == len(occ_class_names) - 2))
    table.add_row(['MEAN', rayiou_0.item(), rayiou_1.item(), rayiou_2.item()])
    print(table)

    per_class = {}
    for i in range(len(occ_class_names) - 1):
        per_class[occ_class_names[i]] = {
            'RayIoU@1': iou_list[0][i].item() if torch.isfinite(iou_list[0][i]) else float('nan'),
            'RayIoU@2': iou_list[1][i].item() if torch.isfinite(iou_list[1][i]) else float('nan'),
            'RayIoU@4': iou_list[2][i].item() if torch.isfinite(iou_list[2][i]) else float('nan'),
        }
    return {
        'RayIoU': rayiou.item(),
        'RayIoU@1': rayiou_0.item(),
        'RayIoU@2': rayiou_1.item(),
        'RayIoU@4': rayiou_2.item(),
        'per_class': per_class,
        'table': str(table),
    }


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


def eval_rayiou_nucraft_for_scene(
    scene_dir,
    scene,
    data_path,
    mapping_scene,
    gt_path,
    setting="nuCraft",
    model_name="",
    device='cuda',
    print_table=True,
):
    occ_ls = sorted([f for f in os.listdir(join(scene_dir, 'Occ'))])
    if len(occ_ls) < 38:
        raise ValueError(
            f"Not enough occupancy files in {scene_dir}/Occ. Found {len(occ_ls)}, expected at least 38."
        )

    lidar_rays = torch.from_numpy(generate_lidar_rays()).to(device=device, dtype=torch.float32)
    pcd_pred_list, pcd_gt_list = [], []
    free_id = len(OCC_CLASS_NAMES) - 1

    for occ_file in occ_ls:
        ckpt = torch.load(join(scene_dir, 'Occ', occ_file), map_location='cpu')
        timestep = int(occ_file.split('_')[-1].split('.')[0])
        ego2world = np.loadtxt(join(data_path, scene, 'ego', f'{timestep:02d}.txt'))
        lidar2world_ref = np.loadtxt(join(data_path, scene, 'lidar', f'{timestep:02d}.txt'))
        ego2lidar = np.matmul(np.linalg.inv(lidar2world_ref), ego2world)

        tp_ls = []
        for i in range(0, len(occ_ls), 5):
            lidar2world = np.loadtxt(join(data_path, scene, 'lidar', f'{i:02d}.txt'))
            lidar_origin_lidar_ref = get_lidar_origin_in_lidar_ref(lidar2world_ref, lidar2world).reshape(1, 1, 3)
            tp_ls.append(torch.tensor(lidar_origin_lidar_ref, device='cpu'))
        lidar_origin_lidar_ref = torch.cat(tp_ls, dim=1)

        nucraft_gt = load_nucraft_gt(join(gt_path, f"{mapping_scene['LIDAR_TOP'][f'{timestep:0>2d}']}.bin"))
        dense_cls_gt = sparse2dense(*nucraft_gt, *settings['nuCraft_np']).astype(np.uint8)
        dense_cls_gt = torch.tensor(dense_cls_gt, device=device)

        if model_name.startswith('selfocc'):
            dense_cls_pred = transform_occ_from_ego_to_lidar_torch(
                ckpt['voxel_indices'],
                ckpt['voxel_cls'],
                torch.tensor(ego2lidar, device=device, dtype=torch.float32),
            )
            mask = dense_cls_pred == 255
            dense_cls_pred[mask] = 17
            dense_cls_gt[mask] = 17
        else:
            dense_cls_pred = sparse2dense_torch(
                ckpt['voxel_indices'].long(),
                ckpt['voxel_cls'].long(),
                *settings[setting],
            )

        sem_pred, sem_gt = dense_cls_pred, dense_cls_gt
        lidar_origins = lidar_origin_lidar_ref.to(device)
        pcd_pred = process_one_sample(
            sem_pred=sem_pred,
            lidar_rays=lidar_rays,
            output_origin=lidar_origins,
            occ_class_names=OCC_CLASS_NAMES,
        )
        pcd_gt = process_one_sample(
            sem_pred=sem_gt,
            lidar_rays=lidar_rays,
            output_origin=lidar_origins,
            occ_class_names=OCC_CLASS_NAMES,
        )
        valid_mask = (pcd_gt[:, 0].long() != free_id)
        pcd_pred_list.append(pcd_pred[valid_mask])
        pcd_gt_list.append(pcd_gt[valid_mask])
        torch.cuda.empty_cache()

    iou_list = calc_rayiou(pcd_pred_list, pcd_gt_list, OCC_CLASS_NAMES)
    result = _format_rayiou_result(iou_list)
    return result


def eval_rayiou_occ3d_for_scene(
    scene_dir,
    scene,
    data_path,
    mapping_scene,
    gt_path,
    setting="Occ3D",
    print_table=True,
):
    occ_ls = sorted([f for f in os.listdir(join(scene_dir, 'Occ'))])
    if len(occ_ls) < 38:
        raise ValueError(
            f"Not enough occupancy files in {scene_dir}/Occ. Found {len(occ_ls)}, expected at least 38."
        )

    lidar_origin_ego_ls = []
    gt_dense_occ_ls = []
    pd_dense_occ_ls = []

    for occ_file in occ_ls:
        ckpt = torch.load(join(scene_dir, 'Occ', occ_file), map_location='cpu')
        timestep = int(occ_file.split('_')[-1].split('.')[0])
        tp_ls = []
        ego2world = np.loadtxt(join(data_path, scene, 'ego', f'{timestep:02d}.txt'))
        for i in range(0, len(occ_ls), 5):
            lidar2world = np.loadtxt(join(data_path, scene, 'lidar', f'{i:02d}.txt'))
            lidar_origin_ego = lidar_origin_in_ego(ego2world, lidar2world).reshape(1, 1, 3)
            tp_ls.append(torch.tensor(lidar_origin_ego, device='cpu'))
        lidar_origin_ego = torch.cat(tp_ls, dim=1)

        dense_cls_occ, mask = load_occ3d_gt(
            join(gt_path, scene, f"{mapping_scene['sample_token'][f'{timestep:0>2d}']}/labels.npz")
        )
        dense_cls_occ[~mask] = 17
        dense_cls_occ = dense_cls_occ.astype(np.uint8)

        dense_cls_from_gs = sparse2dense_torch(
            ckpt['voxel_indices'].long(),
            ckpt['voxel_cls'].long(),
            *settings[setting],
        )
        dense_cls_from_gs[~mask] = 17
        dense_cls_from_gs = dense_cls_from_gs.cpu().numpy().astype(np.uint8)

        lidar_origin_ego_ls.append(lidar_origin_ego)
        gt_dense_occ_ls.append(dense_cls_occ)
        pd_dense_occ_ls.append(dense_cls_from_gs)

    metrics = main_rayiou(
        pd_dense_occ_ls,
        gt_dense_occ_ls,
        lidar_origin_ego_ls,
        occ_class_names=OCC_CLASS_NAMES,
    )
    if not print_table:
        pass
    return metrics
