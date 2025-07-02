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
parser.add_argument('--gt_path', type=str, default='/media/fengyi/bb/nuCraft/occupancy@0.2/')
parser.add_argument('--dataset', type=str, default='nuCraft')
args = parser.parse_args(sys.argv[1:])
data_path = args.data_path
gt_path = args.gt_path
dataset = args.dataset

mapping = json.load(open(f"{data_path}/mapping.json"))
vis = VoxelGridVisualizer('Comparison', 1, 600, dataset)
model_ls = [
    f'out-main-{dataset}/depth-E',
    f'out-main-{dataset}/lidar-E',
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
        
            mask = ckpt['voxel_cls'] != 0
            timestep = int(occ_file.split('_')[-1].split('.')[0])
            ckpt['voxel_indices'] = ckpt['voxel_indices'][mask]
            ckpt['voxel_cls'] = ckpt['voxel_cls'][mask]
            view_ls.append((ckpt['voxel_indices'].cpu().numpy(), ckpt['voxel_cls'].cpu().numpy()))


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