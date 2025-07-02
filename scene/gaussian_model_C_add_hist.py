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

import numpy as np
import torch
from torch import nn
from scene.gaussian_model_B_scale_aware_voxel import GaussianModel as GaussianModel_Parent
import ipdb
import time

class GaussianModel(GaussianModel_Parent):
    
    def add_from_pcd(self, train_data, *args, **kwargs):
        self.create_from_pcd(train_data)
        self._xyz = nn.Parameter(torch.cat((self._xyz, self.detach_dict['_xyz']), dim=0).requires_grad_(True))
        self._rgb = nn.Parameter(torch.cat((self._rgb, self.detach_dict['_rgb']), dim=0).requires_grad_(True))
        self._opacity = nn.Parameter(torch.cat((self._opacity, self.detach_dict['_opacity']), dim=0).requires_grad_(True))
        self._scaling = nn.Parameter(torch.cat((self._scaling, self.detach_dict['_scaling']), dim=0).requires_grad_(True))
        self._rotation = nn.Parameter(torch.cat((self._rotation, self.detach_dict['_rotation']), dim=0).requires_grad_(True))
        self._semantic = torch.cat((self._semantic, self.detach_dict['_semantic']), dim=0)