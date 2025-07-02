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

import ipdb
import torch
import os
import random
import numpy as np
import sys
from gaussian_renderer import render, network_gui


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

os.environ['PYTHONHASHSEED'] = '42'

from utils.loss_utils import l1_loss, ssim
from scene import *
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, num_classes

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor

from multiprocessing import Pool

import time
import torch
import matplotlib.pyplot as plt
import numpy as np

def training(model_params, optimization_params, pipeline_params, debug_from, model):
    training_start = time.time()
    if not SPARSE_ADAM_AVAILABLE and optimization_params.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")
    tb_writer = prepare_output_and_logger(model_params)

    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    # bg_color = [1 for _ in range(20)] if dataset.white_background else [0 for _ in range(20)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = eval(model)(optimization_params.optimizer_type)
    scene = Scene(model_params, optimization_params, pipeline_params, gaussians, background)
    total_frame = scene.total_frame

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = optimization_params.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 

    depth_colormap = plt.get_cmap('plasma')  # 'viridis', 'jet', 'plasma', 'inferno'
    bg = torch.rand((3), device="cuda") if optimization_params.random_background else background
    with ThreadPoolExecutor() as executor:
        progress_bar = tqdm(range(total_frame), desc="Frame Progress")
        for frame_id in progress_bar:
            scene.update_scene(frame_id)
            gaussians.training_setup(optimization_params)
            
            training_start = time.time()
            for iteration in range(1, scene.len_train_cameras + 1):
                if network_gui.conn == None:
                    network_gui.try_connect()
                while network_gui.conn != None:
                    try:
                        net_image_bytes = None
                        custom_cam, do_training, depth, semantic, keep_alive, scaling_modifier = network_gui.receive()
                        if custom_cam != None:
                            render_pkg = render(custom_cam, gaussians, pipeline_params, background, semantic, scaling_modifier=scaling_modifier) 
                            if depth:
                                net_image = process_inverse_depth_map(render_pkg["depth"]).cpu().numpy()
                                net_image /= net_image.max()
                                net_image = (depth_colormap(net_image)[:, :, :3] * 255).astype(np.uint8)
                            elif semantic:
                                net_image = gaussians.colors[torch.argmax(render_pkg["semantic"], dim=0).cpu().numpy()]
                            else:
                                net_image = (torch.clamp(render_pkg["render"], min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                            net_image_bytes = memoryview(net_image)
                        network_gui.send(net_image_bytes, model_params.source_path)
                        if do_training and ((iteration < scene.len_train_cameras + 1) or not keep_alive):
                            break
                    except Exception as e:
                        network_gui.conn = None
                        
                iter_start.record()

                viewpoint_cam = scene.getTrainCameras(iteration-1)
                if (iteration - 1) == debug_from:
                    pipeline_params.debug = True
                
                render_pkg = render(viewpoint_cam, gaussians, pipeline_params, bg)
                image = render_pkg["render"]
                radii = render_pkg["radii"]

                if viewpoint_cam.alpha_mask is not None:
                    alpha_mask = viewpoint_cam.alpha_mask
                    image *= alpha_mask

                # Loss
                try:
                    gt_semantic = viewpoint_cam.semantic.long() # [1, H, W]
                    non_sky_mask = gt_semantic != 17 # [1, H, W]
                except Exception as e:
                    non_sky_mask = torch.ones_like(image)
                
                gt_image = viewpoint_cam.original_image
                # *non_sky_mask # [3, H, W]
                Ll1 = l1_loss(image, gt_image)
                loss = Ll1

                loss.backward()

                iter_end.record()

                with torch.no_grad():
                    # Log and save
                    training_report(tb_writer, iteration, Ll1, loss, iter_start.elapsed_time(iter_end))

                    # Optimizer step
                    if iteration < optimization_params.iterations:
                        if use_sparse_adam:
                            visible = radii > 0
                            gaussians.optimizer.step(visible, radii.shape[0])
                            gaussians.optimizer.zero_grad(set_to_none = True)
                        else:
                            gaussians.optimizer.step()
                            gaussians.optimizer.zero_grad(set_to_none = True)
            scene.training_time += time.time() - training_start
            # print(f"Training time: {time.time() - training_start:.3f}\n-----------------------------------")
            scene.save_eval(frame_id)
            # ThreadPoolExecutor().submit(scene.fusion_save_eval, frame_id)
        progress_bar.close()
        scene.report_results()   

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, elapsed):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--model", type=str)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), 
             args.debug_from, args.model)
