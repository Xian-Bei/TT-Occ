from datetime import datetime
import os
import shutil
import ipdb
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(1)
import sys
pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)
from detectron2.data import MetadataCatalog
import torch
import time
import logging

from torchvision import transforms

from utils.arguments import load_opt_command

from openseed.BaseModel import BaseModel
from openseed import build_model

logger = logging.getLogger(__name__)
import threading

thing_dic = {
    'bicycle': 2, # 0
    'bus': 3, 
    'car': 4, 
    'sedan': 4,
    'van': 4,
    'construction vehicle': 5, 
    'crane': 5,
    'excavator': 5,
    'motorcycle': 6,
    'person': 7, 
    'pedestrian': 7,
    'truck': 10,
}   
stuff_dic = {
    
    'traffic cone': 8,
    'cone': 8,
    'road': 11,   
    'highway': 11,
    'street': 11,
    'sidewalk': 13,
    'terrain': 14,
    'grass': 14,
    'building': 15, 
    'wall': 15,
    'fence': 15,
    'bridge': 15,
    'pole': 15,
    'traffic pole': 15,
    'traffic light': 15,
    'traffic sign': 15,
    'street sign': 15,
    'street pole': 15,
    'streetlight': 15,
    'hydrant': 15,
    'meter box': 15,
    'display window': 15,
    'skyscraper': 15,
    'parking meter': 15,
    'tower': 15,
    'house': 15,
    'structure': 15,
    'banner': 15,
    'board': 15,
    'billboard': 15,
    'stairs': 15,
    'pillar': 15,
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

   
if __name__ == "__main__":
    
    
    opt, cmdline_args = load_opt_command(None)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['user_dir'] = absolute_user_dir

    nus_dir = opt['user_dir']
    scene_list = [scene for scene in sorted(os.listdir(nus_dir)) if scene.startswith('scene')]

    pretrained_pth = os.path.join(opt['WEIGHT'])

    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)
 
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes + stuff_classes, is_eval=False)
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(thing_classes + stuff_classes)

    save_vis = True
    batch_size = 1
    with torch.no_grad():
        for scene_id, scene in enumerate(scene_list):
            for cam in [0, 1, 2, 3, 4, 5]:
                print(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"), scene_id, scene, cam)
                image_dir = os.path.join(nus_dir, scene, f'{cam}')

                output_root = os.path.join(nus_dir, scene, f'swint_{cam}')
                if os.path.exists(output_root):
                    shutil.rmtree(output_root)
                
                output_root = os.path.join(nus_dir, scene, f'openseed_{cam}')
                if os.path.exists(output_root):
                    shutil.rmtree(output_root)
                os.makedirs(output_root, exist_ok=True)
                    
                if save_vis:
                    vis_dir = os.path.join(output_root,'vis')
                    os.makedirs(vis_dir, exist_ok=True)
                batch_inputs = []
                image_list = [image_name for image_name in sorted(os.listdir(image_dir)) if image_name.endswith('.jpg')]
                start_time = time.time()
                for image_idx, image_name in enumerate(image_list):
                    image_pth = os.path.join(image_dir, image_name)
                    image_ori = Image.open(image_pth).convert("RGB")
                    width = image_ori.size[0]
                    height = image_ori.size[1]
                    image = transform(image_ori)
                    image = np.asarray(image)
                    image_ori = np.asarray(image_ori)
                    images = torch.from_numpy(image).permute(2,0,1).cuda()
                    batch_inputs.append({'image': images, 'height': height, 'width': width, 'image_name': image_name.split('.')[0], 'image_ori': image_ori})
                    if len(batch_inputs) == batch_size or image_idx == len(image_list)-1:
                        outputs = model.forward(batch_inputs)
                        threading.Thread(target=save, args=(outputs, batch_inputs, output_root, save_vis, vis_dir), daemon=False).start()
                        # save(outputs, batch_inputs, output_root, save_vis, vis_dir)

                        batch_inputs = []
                print(f'average time: {(time.time()-start_time)/len(image_list)*1000*6:.0f}ms')
                torch.cuda.empty_cache()
