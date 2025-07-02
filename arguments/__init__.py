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

from argparse import ArgumentParser, Namespace
import sys
import os
import numpy as np
import torch

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


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
    
    # [200, 200, 200],# Black             # 0 'noise/ignore' 
    # [112, 128, 144],# Slategrey         # 1 'barrier',              
    # [220, 20, 60],  # Crimson       # 2 'bicycle',              
    # [255, 127, 80], # Coral         # 3 'bus',                  
    # [255, 158, 0],  # Orange        # 4 'car',                  
    # [233, 150, 70], # Darksalmon    # 5 'construction_vehicle', 
    # [255, 61, 99],  # Red           # 6 'motorcycle',           
    # [0, 0, 230],    # Blue          # 7 'pedestrian',           
    # [47, 79, 79],   # Darkslategrey     # 8 'traffic_cone',         
    # [255, 140, 0],  # Darkorange    # 9 'trailer',              
    # [255, 99, 71],  # Tomato        # 10'truck',                
    # [0, 207, 191],  # nuTonomygreen     # 11'driveable_surface',    
    # (70, 130, 180), # Steelblue,        # 12'otherflat'
    # [75, 0, 75],    # purple            # 13'sidewalk',             
    # [112, 180, 60],                     # 14'terrain',              
    # [222, 184, 135],# Burlywood         # 15'manmade',              
    # [0, 175, 0],    # Green             # 16'vegetation',           
    # [135, 206, 235],# Skyblue       # 17'sky', 
], dtype=np.uint8)
colors_1 = colors/255
num_classes = len(colors)-1 # sky is not counted
empty_id = 17  
semantic_list = [ # 15 classes without 0, 12, 17
    # 'ignore',
    'barrier',
    'bicycle',
    'bus',
    'car',
    'construction_vehicle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'trailer',
    'truck',
    'driveable_surface',
    # 'otherflat',
    'sidewalk',
    'terrain',
    'manmade',
    'vegetation',
    # 'empty'
]
settings = {
    "Occ3D_np": [np.array([[-40., -40., -1.]]), np.array([[40., 40., 5.4]]), 0.4],
    "Occ3D": [torch.tensor([[-40., -40., -1.]], device='cuda'), torch.tensor([[40., 40., 5.4]], device='cuda'), 0.4],
    "nuCraft_np": [np.array([[-51.2, -51.2, -5]]), np.array([[51.2, 51.2, 3]]), 0.2],
    "nuCraft": [torch.tensor([[-51.2, -51.2, -5]], device='cuda'), torch.tensor([[51.2, 51.2, 3]], device='cuda'), 0.2]
}
ego_range = {
    'w': 1.73,
    'l': 4.08,
    'h': 1.84,
    'eps': 0.1,
}
class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 0
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        
        self._white_background = True

        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = True
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.position_lr_init = 0.00016
        self.scaling_lr = 0.005
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.random_background = False
        self.optimizer_type = "sparse_adam"
        self.densification_interval = 100

        
        self.occ_setting = "Occ3D"
        # self.occ_setting = "nuCraft"
        self.occ3d_path = ""
        self.nucraft_path = ""
        self.save_ply = False
        self.save_occ = True
        self.eval_occ = False
        self.vis_occ = False
        self.multi_thread = False
        self.iterations = 0
        self.display_iter = 1
        self.semantic_weight = 0
        self.K = 10
        self.variant = 'lidar'
        
        super().__init__(parser, "Optimization Parameters")
        

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
