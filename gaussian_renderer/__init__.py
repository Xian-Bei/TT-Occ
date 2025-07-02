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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_semantic import GaussianRasterizer as GaussianRasterizer_semantic
from utils.sh_utils import eval_sh, SH2RGB

def render(viewpoint_camera, pc, pipe, bg_color, semantics=False, scaling_modifier = 1.0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    means3D = pc.get_xyz
    screenspace_points = torch.zeros_like(means3D, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    means2D = screenspace_points
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )
    if semantics:
        rasterizer = GaussianRasterizer_semantic(raster_settings=raster_settings)
    else:
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pc.compute_cov3D_python:
        raise
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    if scales.shape[1] == 1:
        scales = scales.repeat(1, 3)
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    dc = None
    shs = None
    colors_precomp = None
    colors_precomp = pc.get_rgb
    if semantics:
        colors_precomp = torch.cat([colors_precomp, pc.get_semantic], dim=1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth_image = rasterizer(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
        
    rendered_image, rendered_semantic = rendered_image[:3, :, :], rendered_image[3:, :, :]
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image,
        "semantic" : rendered_semantic 
        }
    
    return out
