from datetime import datetime
import os
import shutil
import subprocess
import ipdb
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(1)
import sys
pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

# Pillow>=10 removed Image.LINEAR; detectron2 0.6 still references it (same as BILINEAR).
from PIL import Image
if not hasattr(Image, "LINEAR"):
    Image.LINEAR = Image.BILINEAR

from detectron2.data import MetadataCatalog
import torch
import time
import logging

from torchvision import transforms
from tqdm import tqdm

try:
    from utils.arguments import load_opt_command
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    _args_path = Path(__file__).resolve().parent / "utils" / "arguments.py"
    _spec = importlib.util.spec_from_file_location("openseed_utils_arguments", _args_path)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    load_opt_command = _module.load_opt_command

from openseed.BaseModel import BaseModel
from openseed import build_model

logger = logging.getLogger(__name__)
import threading

thing_dic = {
    'bicycle': 2, # 0
    'bus': 3, 
    'car': 4, 
    'construction_vehicle': 5, 
    'crane': 5,
    'motorcycle': 6,
    'person': 7, 
    'trailer': 9,
    'trailer_truck': 9,
    'truck': 10,
    
}   
stuff_dic = {
    'barrier': 1,
    'traffic_cone': 8,
    'road': 11,   
    'sidewalk': 13,
    'terrain': 14,
    'grass': 14,
    
    'building': 15, 
    'wall': 15,
    'tree': 16,
    'sky': 17
}


# colors = np.array([
#     [0, 0, 0],      # Black         # 0 Others
#     [112, 128, 144],# Slategrey     # 1 'barrier',              
#     [220, 20, 60],  # Crimson       # 2 'bicycle',              
#     [255, 127, 80], # Coral         # 3 'bus',                  
#     [255, 158, 0],  # Orange        # 4 'car',                  
#     [233, 150, 70], # Darksalmon    # 5 'construction_vehicle', 
#     [255, 61, 99],  # Red           # 6 'motorcycle',           
#     [0, 0, 230],    # Blue          # 7 'pedestrian',           
#     [47, 79, 79],   # Darkslategrey # 8 'traffic_cone',         
#     [255, 140, 0],  # Darkorange    # 9 'trailer',              
#     [255, 99, 71],  # Tomato        # 10'truck',                
#     [0, 207, 191],  # nuTonomygreen # 11'driveable_surface',
#     (70, 130, 180), # Steelblue,    # 12 None
#     [75, 0, 75],    # purple        # 13'sidewalk',             
#     [112, 180, 60],                 # 14'terrain',              
#     [222, 184, 135],# Burlywood     # 15'manmade',              
#     [0, 175, 0],    # Green         # 16'vegetation',           
#     [135, 206, 235],# Skyblue       # 17'sky',                  
# ], dtype=np.uint8)

colors = np.array([
    [200, 200, 200],       # 0 noise                  black
    [255, 120,  50],       # 1 barrier              orange
    [255, 192, 203],       # 2 bicycle              pink
    [255, 255,   0],       # 3 bus                  yellow
    [  0, 150, 245],       # 4 car                  blue
    [  0, 255, 255],       # 5 construction_vehicle cyan
    [255, 127,   0],       # 6 motorcycle           dark orange
    [255,   0,   0],       # 7 pedestrian           red
    [255, 240, 150],       # 8 traffic_cone         light yellow
    [135,  60,   0],       # 9 trailer              brown
    [160,  32, 240],       # 10 truck                purple                
    [255,   0, 255],       # 11 driveable_surface    dark pink
    [175,   0,  75],       # 12 other_flat           dark red
    [ 75,   0,  75],       # 13 sidewalk             dard purple
    [150, 240,  80],       # 14 terrain              light green          
    [230, 230, 250],       # 15 manmade              white
    [  0, 175,   0],       # 16 vegetation           green
    [0 ,  0  , 0  ],       # 17 sky                  black
], dtype=np.uint8)
import cv2
from PIL import Image

mapping = np.array([v for k, v in {**thing_dic, **stuff_dic}.items()])
thing_classes = list(thing_dic.keys())
stuff_classes = list(stuff_dic.keys())
thing_dataset_id_to_contiguous_id = {x:x for x in range(len(thing_classes))}
stuff_dataset_id_to_contiguous_id = {x+len(thing_classes):x for x in range(len(stuff_classes))}
thing_colors = [colors[v].tolist() for v in thing_dic.values()]
stuff_colors = [colors[v].tolist() for v in stuff_dic.values()]
MetadataCatalog.get("demo").set(
    thing_colors=thing_colors,
    thing_classes=thing_classes,
    thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
    stuff_colors=stuff_colors,
    stuff_classes=stuff_classes,
    stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
)
metadata = MetadataCatalog.get('demo')


def calculate_bounding_box(mask):
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    return (x_min, y_min, x_max, y_max)

def check_adjacent(mask1, mask2, threshold=10):
    kernel = np.ones((3, 3), np.uint8)
    mask1_dilated = cv2.dilate(mask1.astype(np.uint8), kernel, iterations=threshold)
    return np.any(mask1_dilated & mask2)

def check_position_relation(bbox_person, bbox_bicycle):

    y_person_center = (bbox_person[1] + bbox_person[3]) / 2
    y_bicycle_center = (bbox_bicycle[1] + bbox_bicycle[3]) / 2
    vertical_check = y_person_center <= y_bicycle_center
    
    x_person_center = (bbox_person[0] + bbox_person[2]) / 2
    horizontal_check = bbox_bicycle[0] < x_person_center < bbox_bicycle[2]
    # x_bicycle_center = (bbox_bicycle[0] + bbox_bicycle[2]) / 2
    # horizontal_check = abs(x_person_center - x_bicycle_center) < (bbox_bicycle[2] - bbox_bicycle[0]) * 0.5

    return vertical_check and horizontal_check

def is_person_riding_bicycle(mask_bicycle, mask_person):
    bbox_person = calculate_bounding_box(mask_person)
    bbox_bicycle = calculate_bounding_box(mask_bicycle)

    adjacent = check_adjacent(mask_person, mask_bicycle)
    position_relation = check_position_relation(bbox_person, bbox_bicycle)
    return adjacent and position_relation

# {'id': int, 'isthing': bool, 'category_id': int}
def merge_bicycle_motorcycle(img_ret_id, info_ret):
    bicycle_areas = [(x['id'], img_ret_id == x['id']) for x in info_ret if x['isthing'] and mapping[x['category_id']] == 2]
    motorcycle_areas = [(x['id'], img_ret_id == x['id']) for x in info_ret if x['isthing'] and mapping[x['category_id']] == 6]
    person_areas = [(x['id'], img_ret_id == x['id']) for x in info_ret if x['isthing'] and mapping[x['category_id']] == 7]
    person_used = [False] * len(person_areas)
    for b_i, bicycle_area in bicycle_areas + motorcycle_areas:
        for p_ii, (p_i, person_area) in enumerate(person_areas):
            if not person_used[p_ii] and is_person_riding_bicycle(bicycle_area, person_area):
                # print('merge a person into a bicycle')
                person_used[p_ii] = True
                img_ret_id[img_ret_id==p_i] = b_i
                # person_id_to_delete.append(p_i)
    return img_ret_id, info_ret

def vis(img_ret_id, info_ret):
    info_category_id = mapping[np.array([x['category_id'] for x in info_ret])]
    info_category_id = np.array([0] + info_category_id.tolist()).astype(np.uint8)
    if 1 in info_category_id or 9 in info_category_id:
        print("#############################################################################")
    img_category_id = info_category_id[img_ret_id]
    img_category_id_vis = colors[img_category_id]
    # img_category_id_vis = Image.fromarray(img_category_id_vis)
    return img_category_id, img_category_id_vis

def save(outputs, batch_inputs, output_root, save_vis, vis_dir):
    for i, output in enumerate(outputs):
        pano_seg = output['panoptic_seg'][0].cpu().numpy() # (h, w)
        pano_seg_info = output['panoptic_seg'][1] # list of dict, each dict is {'id': int, 'isthing': bool, 'category_id': int}
        if len(pano_seg_info) > 0:
            pano_seg, pano_seg_info = merge_bicycle_motorcycle(pano_seg, pano_seg_info)
            res, res_vis = vis(pano_seg, pano_seg_info)
            Image.fromarray(res).save(os.path.join(output_root, f'{batch_inputs[i]["image_name"]}.png'))
            for j in range(len(pano_seg_info)):
                if pano_seg_info[j]['isthing']:
                    continue
                else:
                    pano_seg[pano_seg == pano_seg_info[j]['id']] = 0
            #     plt.imshow(pano_seg == j)
            #     plt.show()
            cv2.imwrite(os.path.join(output_root, f'{batch_inputs[i]["image_name"]}_instance.png'), pano_seg.astype(np.uint8))
            if save_vis:
                Image.fromarray(res_vis).save(os.path.join(vis_dir, f'{batch_inputs[i]["image_name"]}_vis.png'))


def get_own_gpu_memory(gpu=0):
    pid = os.getpid()
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,nounits,noheader"
        ]
    )
    for line in result.decode().strip().split('\n'):
        proc_pid, mem = line.split(',')
        if int(proc_pid.strip()) == pid:
            return int(mem.strip())  # 单位MB
    return 0  # 如果没查到

def get_inference_context(use_inference_mode=True):
    return torch.inference_mode if use_inference_mode else torch.no_grad


def get_autocast_context():
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    return torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype)

def _resolve_weight_path(weight_path):
    if os.path.isabs(weight_path) and os.path.exists(weight_path):
        return weight_path
    candidates = [
        os.path.abspath(weight_path),
        os.path.abspath(os.path.join(os.getcwd(), weight_path)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), weight_path)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.basename(weight_path))),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return weight_path

def build_openseed_runtime(conf_file, weight_path, use_inference_mode=True):
    weight_path = _resolve_weight_path(weight_path)
    opt, _ = load_opt_command(['evaluate', '--conf_files', conf_file, '--overrides', 'WEIGHT', weight_path])
    pretrained_pth = _resolve_weight_path(opt['WEIGHT'])
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    t = [transforms.Resize(512, interpolation=Image.BICUBIC)]
    transform = transforms.Compose(t)
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
        thing_classes + stuff_classes, is_eval=False
    )
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(thing_classes + stuff_classes)
    return {
        "model": model,
        "transform": transform,
        "use_inference_mode": use_inference_mode,
    }

def process_scene(runtime, scene_in, scene_out, save_vis=False):
    model = runtime["model"]
    transform = runtime["transform"]
    use_inference_mode = runtime["use_inference_mode"]
    peak = 0
    batch_size = 1
    inference_ctx = get_inference_context(use_inference_mode)
    with inference_ctx():
        for cam in tqdm([0, 1, 2, 3, 4, 5], desc="OpenSeeD cams", leave=False):
            image_dir = os.path.join(scene_in, f'{cam}')
            cam_output_root = os.path.join(scene_out, f'openseed_{cam}')
            os.makedirs(cam_output_root, exist_ok=True)
            vis_dir = None
            if save_vis:
                vis_dir = os.path.join(cam_output_root, 'vis')
                os.makedirs(vis_dir, exist_ok=True)
            batch_inputs = []
            image_list = [image_name for image_name in sorted(os.listdir(image_dir)) if image_name.endswith('.jpg')]
            for image_idx, image_name in enumerate(
                tqdm(image_list, desc=f"OpenSeeD cam{cam}", leave=False)
            ):
                image_pth = os.path.join(image_dir, image_name)
                image_ori = Image.open(image_pth).convert("RGB")
                width = image_ori.size[0]
                height = image_ori.size[1]
                image = transform(image_ori)
                image = np.asarray(image)
                image_ori = np.asarray(image_ori)
                images = torch.from_numpy(image).permute(2, 0, 1).cuda()
                batch_inputs.append(
                    {
                        'image': images,
                        'height': height,
                        'width': width,
                        'image_name': image_name.split('.')[0],
                        'image_ori': image_ori,
                    }
                )
                if len(batch_inputs) == batch_size or image_idx == len(image_list) - 1:
                    with get_autocast_context():
                        outputs = model.forward(batch_inputs)
                    save(outputs, batch_inputs, cam_output_root, save_vis, vis_dir)
                    batch_inputs = []
                peak = max(peak, get_own_gpu_memory())
    return {"peak_mb": peak}


def _run_single_prediction(output):
    pano_seg = output["panoptic_seg"][0].cpu().numpy()
    pano_seg_info = output["panoptic_seg"][1]
    if len(pano_seg_info) > 0:
        pano_seg, pano_seg_info = merge_bicycle_motorcycle(pano_seg, pano_seg_info)
        semantic_map, semantic_vis = vis(pano_seg, pano_seg_info)
        for seg_info in pano_seg_info:
            if not seg_info["isthing"]:
                pano_seg[pano_seg == seg_info["id"]] = 0
        instance_map = pano_seg.astype(np.uint8)
    else:
        semantic_map = np.zeros_like(pano_seg, dtype=np.uint8)
        semantic_vis = colors[semantic_map]
        instance_map = np.zeros_like(pano_seg, dtype=np.uint8)
    return semantic_map.astype(np.uint8), instance_map, semantic_vis


def process_frame_openseed(runtime, scene_in, scene_out, frame, save_vis=False, write_disk=True):
    model = runtime["model"]
    transform = runtime["transform"]
    use_inference_mode = runtime["use_inference_mode"]
    inference_ctx = get_inference_context(use_inference_mode)
    frame_outputs = {}
    frame_name = f"{frame:0>2d}"
    with inference_ctx():
        for cam in range(6):
            image_pth = os.path.join(scene_in, f"{cam}", f"{frame_name}.jpg")
            image_ori = Image.open(image_pth).convert("RGB")
            width, height = image_ori.size
            image = transform(image_ori)
            image = np.asarray(image)
            image_ori = np.asarray(image_ori)
            batch_inputs = [{
                "image": torch.from_numpy(image).permute(2, 0, 1).cuda(),
                "height": height,
                "width": width,
                "image_name": frame_name,
                "image_ori": image_ori,
            }]
            with get_autocast_context():
                outputs = model.forward(batch_inputs)
            semantic_map, instance_map, semantic_vis = _run_single_prediction(outputs[0])
            frame_outputs[cam] = {
                "semantic": semantic_map,
                "instance": instance_map,
            }
            if write_disk:
                cam_output_root = os.path.join(scene_out, f"openseed_{cam}")
                os.makedirs(cam_output_root, exist_ok=True)
                Image.fromarray(semantic_map).save(os.path.join(cam_output_root, f"{frame_name}.png"))
                Image.fromarray(instance_map).save(os.path.join(cam_output_root, f"{frame_name}_instance.png"))
                if save_vis:
                    vis_dir = os.path.join(cam_output_root, "vis")
                    os.makedirs(vis_dir, exist_ok=True)
                    Image.fromarray(semantic_vis).save(os.path.join(vis_dir, f"{frame_name}_vis.png"))
    return frame_outputs

if __name__ == "__main__":
    opt, cmdline_args = load_opt_command(None)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['user_dir'] = absolute_user_dir

    nus_dir = opt['user_dir']
    if cmdline_args.output_dir:
        output_root = os.path.abspath(os.path.expanduser(cmdline_args.output_dir))
    else:
        output_root = nus_dir
    scene_list = [scene for scene in sorted(os.listdir(nus_dir)) if scene.startswith('scene')]

    runtime = build_openseed_runtime(
        conf_file=cmdline_args.conf_files[0],
        weight_path=opt['WEIGHT'],
        use_inference_mode=False,
    )
    save_vis = True
    for scene_id, scene in enumerate(scene_list):
        print(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"), scene_id, scene)
        scene_out = os.path.join(output_root, scene)
        if os.path.exists(scene_out):
            shutil.rmtree(scene_out)
        os.makedirs(scene_out, exist_ok=True)
        start_time = time.time()
        stats = process_scene(runtime, scene_in=os.path.join(nus_dir, scene), scene_out=scene_out, save_vis=save_vis)
        num_images = len([f for f in os.listdir(os.path.join(nus_dir, scene, "0")) if f.endswith(".jpg")]) * 6
        print(f'average time: {(time.time()-start_time)/max(1, num_images)*1000:.0f}ms')
        print(stats["peak_mb"], "MB")