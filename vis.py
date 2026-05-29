import os
import json
from os.path import join
import ipdb
from arguments import semantic_list
import numpy as np
from custom_utils import *
import sys


from argparse import ArgumentParser
parser = ArgumentParser(description="Training script parameters")
parser.add_argument('--data_path', type=str, default="/home/fengyi/Data/new_extracted_nuscenes_val")

# /media/fengyi/bb/Occ3D_nuscenes/gts/
# /media/fengyi/bb/nuCraft/occupancy@0.2/
parser.add_argument('--gt_path', type=str, default='/media/fengyi/bb/Occ3D_nuscenes/gts/')
parser.add_argument('--dataset', type=str, default='Occ3D')

# parser.add_argument('--gt_path', type=str, default='/media/fengyi/bb/Occ3D_nuscenes/gts/')
# parser.add_argument('--dataset', type=str, default='Occ3D')

args = parser.parse_args(sys.argv[1:])
data_path = args.data_path
gt_path = args.gt_path
dataset = args.dataset

mapping = json.load(open(f"{data_path}/mapping.json"))
vis = VoxelGridVisualizer('Comparison', 1, 600, dataset)
model_ls = [
    # f'tp_selfocc/nucraft/selfocc_2',
    # f'tp_selfocc/occ3d_all/selfocc',
    'out-main-Occ3D/lidar-B',
    
]

for scene in sorted(os.listdir(model_ls[0]))[0:]:
    scene_dir = join(model_ls[0], scene)
    if not os.path.isdir(scene_dir):
        continue
    occ_ls = sorted([file for file in os.listdir(join(scene_dir, 'Occ'))])
    for occ_file in occ_ls:
        print("visualizing", scene, 'for', occ_file)
        time = occ_file.split('_')[-1].split('.')[0]
        view_ls = []
        for model in model_ls:
            scene_dir = join(model, scene)
            ckpt = torch.load(join(scene_dir, 'Occ', occ_file))
        
            timestep = int(occ_file.split('_')[-1].split('.')[0])
            
            
            ego2world = np.loadtxt(join(data_path, scene, 'ego', f'{timestep:02d}.txt'))
            lidar2world_ref = np.loadtxt(join(data_path, scene, 'lidar', f'{timestep:02d}.txt'))
            ego2lidar = np.matmul(np.linalg.inv(lidar2world_ref), ego2world)

            dense_cls_from_gs = transform_occ_from_ego_to_lidar(ckpt['voxel_indices'].cpu().numpy(), ckpt['voxel_cls'].cpu().numpy(), ego2lidar)
            cnm_mask = dense_cls_from_gs == 255
            dense_cls_from_gs[cnm_mask] = 17
            ckpt['voxel_indices'], ckpt['voxel_cls'] = dense2sparse(dense_cls_from_gs)

            
            mask = ckpt['voxel_cls'] != 0
            ckpt['voxel_indices'] = ckpt['voxel_indices'][mask]
            ckpt['voxel_cls'] = ckpt['voxel_cls'][mask]

            view_ls.append((ckpt['voxel_indices'], ckpt['voxel_cls']))
            # view_ls.append((ckpt['voxel_indices'].cpu().numpy(), ckpt['voxel_cls'].cpu().numpy()))

        if dataset == 'Occ3D':
            dense_cls_occ_np, cnm_mask = load_occ3d_gt(
                join(gt_path, scene, f"{mapping[scene]['sample_token'][f'{timestep:0>2d}']}/labels.npz"))
            gt_file = dense2sparse(dense_cls_occ_np)
            # view_ls[-1] = (view_ls[-1][0][mask], view_ls[-1][1][mask]) 
            
        else:
            gt_file_path = os.path.join(gt_path, mapping[scene]['LIDAR_TOP'][time]+".bin") 
            gt_file = load_nucraft_gt(gt_file_path)
        view_ls.append(gt_file)


        ego_path = join(data_path, scene, 'ego', time+ '.txt')
        ego2world = np.loadtxt(ego_path)
        lidar_path = join(data_path, scene, 'lidar', time+'.txt') 
        lidar2world = np.loadtxt(lidar_path)
        world2lidar = np.linalg.inv(lidar2world)
        ego2lidar = np.matmul(world2lidar, ego2world)

        view_sparse_voxel(
            view_ls,
            vis,
            f"vis/{scene}_{time}"
        )

print("done")