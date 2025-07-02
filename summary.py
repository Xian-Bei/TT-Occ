import os
import json
from os.path import join
import ipdb
from arguments import semantic_list
import numpy as np
from custom_utils import *
import sys
def format_floats_as_percentage(d):
    for key, value in d.items():
        if isinstance(value, float):
            d[key] = f"{value * 100:.2f}%"
        elif isinstance(value, dict):
            format_floats_as_percentage(value)
    return d

setting = sys.argv[1] 
model_path = sys.argv[2] 
data_path = sys.argv[3]
mapping = json.load(open(f"{data_path}/mapping.json"))
gt_path = sys.argv[4] 

summary = {}
for model in sorted(os.listdir(model_path)):
    model_dir = join(model_path, model)
    if not os.path.isdir(model_dir):
        continue
    summary[model] = {}
    hist_all_scenes = 0
    summary[model]['N'] = 0
    
    for scene in sorted(os.listdir(model_dir)):
        hist_one_scene = 0
        scene_dir = join(model_dir, scene)
        if not os.path.isdir(scene_dir):
            continue
        print("evaluating", scene, 'for', model)
        
        occ_ls = sorted([file for file in os.listdir(join(scene_dir, 'Occ'))])
        if len(occ_ls) < 38:
            raise ValueError(f"Not enough occupancy files in {scene_dir}/Occ. Found {len(occ_ls)}, expected at least 38.")
        for occ_file in occ_ls:
            ckpt = torch.load(join(scene_dir, 'Occ', occ_file))
            timestep = int(occ_file.split('_')[-1].split('.')[0])
            hist_one_frame, _ = eval_occ(timestep, ckpt['voxel_indices'], ckpt['voxel_cls'], setting, scene, mapping[scene], gt_path=gt_path)
            hist_one_scene += hist_one_frame
                
        summary[model]['N'] += 1
        
        iou_camera, miou_camera, mious_camera = cal_iou_miou(hist_one_scene)
        res = format_floats_as_percentage({'iou': iou_camera.item(), 'miou': miou_camera.item(), 'mious': {f'{semantic_list[i]}': mious_camera[i].item() for i in range(len(mious_camera))}})
        json_data = json.dumps(res, indent=4)
        with open(join(scene_dir, f'results.json'), "w") as f:
            f.write(json_data)
            
        hist_all_scenes += hist_one_scene
        
    
    iou_camera, miou_camera, mious_camera = cal_iou_miou(hist_all_scenes)
    summary[model]['Avg_final'] = {'iou': iou_camera.item(), 'miou': miou_camera.item(), 'mious': {f'{semantic_list[i]}': mious_camera[i].item() for i in range(len(mious_camera))}}
    summary[model] = format_floats_as_percentage(summary[model])
    json_data = json.dumps(summary[model], indent=4)
    with open(join(model_dir, f'results_{summary[model]["N"]}.json'), "w") as f:
        f.write(json_data)
    
json_data = json.dumps(summary, indent=4)
with open(join(model_path, f'results.json'), "w") as f:
    f.write(json_data)
print("done")
