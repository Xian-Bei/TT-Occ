from functools import partial
import sys
import time
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import json
import multiprocessing
import shutil
import ipdb
import os
import numpy as np
from nuscenes.nuscenes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

H_orig, W_orig = 900, 1600 
H_target, W_target = 294, 518
new_size = (W_target, H_target)
scale_x = W_target / W_orig
scale_y = H_target / H_orig

def extract(scene, data_root, nusc, save_root):
    print('Processing ', scene['name'])
    cam_list = {          
        "CAM_FRONT": 0,        
        "CAM_FRONT_LEFT": 1,   
        "CAM_BACK_LEFT": 2,    
        "CAM_BACK": 3,          
        "CAM_BACK_RIGHT": 4,   
        "CAM_FRONT_RIGHT": 5,  
    }
    saved_mapping = {}
    saved_mapping['scene_token'] = scene['token']
    saved_mapping.update({'LIDAR_TOP': {}})
    saved_mapping.update({'sample_token': {}})
    scene_dir = os.path.join(save_root, scene['name'])
    ######################################################################
    lidar_dir = os.path.join(scene_dir, 'lidar')
    if os.path.exists(lidar_dir):
        shutil.rmtree(lidar_dir)
    os.makedirs(lidar_dir, exist_ok=True)
    ego_dir = os.path.join(scene_dir, 'ego')
    if os.path.exists(ego_dir):
        shutil.rmtree(ego_dir)
    os.makedirs(ego_dir, exist_ok=True)
    cam_dir_ls = []
    for cam_name, cam_id in cam_list.items():
        cam_dir = os.path.join(scene_dir, f'{cam_id}')
        cam_dir_ls.append(cam_dir)
        if os.path.exists(cam_dir):
            shutil.rmtree(cam_dir)
        os.makedirs(cam_dir, exist_ok=True)
    ######################################################################
    first_sample_token = scene['first_sample_token']
    key_frame_id = 0
    sample_token = first_sample_token
    while sample_token != '':
        # print('Processing ', sample_token)
        sample = nusc.get('sample', sample_token)
        timestamp_mapping = {}
        for sensor_name in ['LIDAR_TOP', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']:
            sensor_token = sample['data'][sensor_name]
            if sensor_name == 'LIDAR_TOP':
                timestamp_mapping.update({'LIDAR_TOP': {}})   
                lidar_data = nusc.get('sample_data', sensor_token)
                saved_mapping['sample_token'][f'{key_frame_id:0>2d}'] = sample_token # 
                saved_mapping['LIDAR_TOP'][f'{key_frame_id:0>2d}'] = lidar_data['token']
                lidar_path = os.path.join(data_root, lidar_data['filename'])
                lidar_points = LidarPointCloud.from_file(lidar_path)
                lidar_save_path = os.path.join(lidar_dir, f'{key_frame_id:0>2d}.bin')
                lidar_points.points.T.astype(np.float32).tofile(lidar_save_path)
                # ######################################################################
                lidar_pose = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
                lidar2ego = transform_matrix(lidar_pose['translation'], Quaternion(lidar_pose['rotation']), inverse=False)
                ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
                ego2world = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
                ego_path = os.path.join(ego_dir, f'{key_frame_id:0>2d}.txt')
                np.savetxt(ego_path, ego2world)
                lidar2world = ego2world @ lidar2ego
                lidar2world_save_path = os.path.join(lidar_dir, f'{key_frame_id:0>2d}.txt')
                np.savetxt(lidar2world_save_path, lidar2world)
            elif sensor_name in cam_list.keys():
                timestamp_mapping.update({cam_list[sensor_name]: {}})   
                cam_data = nusc.get('sample_data', sensor_token)
                cam_dir = cam_dir_ls[cam_list[sensor_name]]
                ######################################################################
                key_path = nusc.get_sample_data_path(cam_data['token'])
                key_save_path = os.path.join(cam_dir, f'{key_frame_id:0>2d}.jpg')
                # image = Image.open(key_path)
                # resized_image = image.resize(new_size)
                # resized_image.save(key_save_path)
                shutil.copy(key_path, key_save_path)
                ######################################################################
                camera_pose = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                cam2ego = transform_matrix(camera_pose['translation'], Quaternion(camera_pose['rotation']), inverse=False)
                ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
                ego2world = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
                cam2world = ego2world @ cam2ego
                cam2world_save_path = os.path.join(cam_dir, f'{key_frame_id:0>2d}.txt')
                np.savetxt(cam2world_save_path, cam2world)
                ######################################################################
                intrinsics = np.array(camera_pose['camera_intrinsic'])
                # intrinsics = intrinsics/downscale
                # intrinsics[2, 2] = 1
                # intrinsics[0, 0] *= scale_x  # fx
                # intrinsics[1, 1] *= scale_y  # fy
                # intrinsics[0, 2] *= scale_x  # cx
                # intrinsics[1, 2] *= scale_y  # cy
                np.savetxt(os.path.join(cam_dir, f'intrinsic.txt'), intrinsics)
                ######################################################################
            
        sample_token = sample['next']
        key_frame_id += 1
    print('Processing done ', scene['name'])
    return saved_mapping

def thread_worker(scene, data_root, nusc, save_root):
    result = extract(scene, data_root=data_root, nusc=nusc, save_root=save_root)
    return scene['name'], result 
    
if __name__ == "__main__":
    split = 'val'
    data_root = '/media/fengyi/bb/nuscenes'
    save_root = f'/home/fengyi/Data/new_extracted_nuscenes_{split}'

    scene_splits = create_splits_scenes()
    nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=True)
    scene_list = [scene for scene in nusc.scene if scene['name'] in scene_splits[split]]

    saved_mapping = {}

    for scene in scene_list:
        saved_mapping[scene['name']] = extract(scene, data_root=data_root, nusc=nusc, save_root=save_root)

    # max_threads = 8
    # futures = {}
    # with ThreadPoolExecutor(max_workers=max_threads) as executor:
    #     for scene in scene_list:
    #         futures.update({executor.submit(thread_worker, scene, data_root, nusc, save_root): scene})
    #         time.sleep(2)
    #     for future in as_completed(futures):
    #         scene_name, result = future.result()  
    #         saved_mapping[scene_name] = result

    with open(os.path.join(save_root, f'mapping.json'), 'w') as f:
        json.dump(saved_mapping, f)