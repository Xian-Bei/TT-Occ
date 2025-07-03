import shutil
import sys
import time
import ipdb
from matplotlib import pyplot as plt

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft.raft import RAFT
from raft.utils import flow_viz
from raft.utils.utils import InputPadder
import torch.nn.functional as F

DEVICE = 'cuda'

def compute_ego_flow_batch(depth_0, K, cam2world_0, cam2world_1):
    """
    Compute ego-motion-induced optical flow in batch mode.

    Inputs:
        depth_0: (B, H, W)
        K: (B, 3, 3)
        cam2world_0: (B, 4, 4)
        cam2world_1: (B, 4, 4)
    Returns:
        flow: (B, 2, H, W)
    """
    B, H, W = depth_0.shape
    device = depth_0.device

    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    u = x.float()[None].expand(B, -1, -1)  # (B, H, W)
    v = y.float()[None].expand(B, -1, -1)

    fx = K[:, 0, 0].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
    fy = K[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = K[:, 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = K[:, 1, 2].unsqueeze(-1).unsqueeze(-1)

    valid_mask = depth_0 > 1
    z = depth_0.clamp(min=1e-3)

    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    pts_cam0 = torch.stack([x_cam, y_cam, z], dim=-1)  # (B, H, W, 3)

    pts_cam0_h = torch.cat([pts_cam0, torch.ones_like(z.unsqueeze(-1))], dim=-1)  # (B, H, W, 4)
    pts_cam0_h = pts_cam0_h.view(B, -1, 4).transpose(1, 2)  # (B, 4, H*W)

    pts_world = cam2world_0 @ pts_cam0_h  # (B, 4, H*W)
    pts_world = pts_world.transpose(1, 2).view(B, H, W, 4)[..., :3]  # (B, H, W, 3)

    world2cam1 = torch.inverse(cam2world_1)  # (B, 4, 4)
    pts_world_h = torch.cat([pts_world, torch.ones_like(z.unsqueeze(-1))], dim=-1)  # (B, H, W, 4)
    pts_world_h = pts_world_h.view(B, -1, 4).transpose(1, 2)  # (B, 4, H*W)

    pts_cam1 = world2cam1 @ pts_world_h  # (B, 4, H*W)
    pts_cam1 = pts_cam1.transpose(1, 2).view(B, H, W, 4)[..., :3]  # (B, H, W, 3)

    x1 = pts_cam1[..., 0]
    y1 = pts_cam1[..., 1]
    z1 = pts_cam1[..., 2]
    
    valid_mask &= z1 > 1
    z1 = z1.clamp(min=1e-3)

    u1 = (x1 * fx.squeeze(-1).squeeze(-1)[:, None, None] / z1) + cx.squeeze(-1).squeeze(-1)[:, None, None]
    v1 = (y1 * fy.squeeze(-1).squeeze(-1)[:, None, None] / z1) + cy.squeeze(-1).squeeze(-1)[:, None, None]

    flow_u = u1 - u
    flow_v = v1 - v
    flow = torch.stack([flow_u, flow_v], dim=1)  # (B, 2, H, W)
    flow = flow.masked_fill(~valid_mask.unsqueeze(1), 0)  # (B, 2, H, W)
    
    return flow


def check_flow_consistency(flow_fwd, flow_bwd, threshold=1.0):
    """
    Perform forward-backward consistency check.

    Args:
        flow_fwd: (B, 2, H, W) forward flow (from t0 to t1)
        flow_bwd: (B, 2, H, W) backward flow (from t1 to t0)
        threshold: float, L2 distance threshold in pixels for consistency

    Returns:
        consistency_mask: (B, H, W) boolean tensor. True = consistent
        fb_error: (B, H, W) float tensor, flow difference magnitude
    """
    B, _, H, W = flow_fwd.shape
    device = flow_fwd.device

    # Generate pixel grid
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    grid = torch.stack([x, y], dim=0).float()  # (2, H, W)
    grid = grid[None].expand(B, -1, -1, -1)  # (B, 2, H, W)

    # Compute destination pixel locations by adding forward flow
    pos_fwd = grid + flow_fwd  # (B, 2, H, W)

    # Normalize grid to [-1, 1] for grid_sample
    pos_fwd_norm = pos_fwd.clone()
    pos_fwd_norm[:, 0] = (pos_fwd_norm[:, 0] / (W - 1)) * 2 - 1  # x
    pos_fwd_norm[:, 1] = (pos_fwd_norm[:, 1] / (H - 1)) * 2 - 1  # y
    pos_fwd_norm = pos_fwd_norm.permute(0, 2, 3, 1)  # (B, H, W, 2)

    # Sample backward flow at warped positions
    flow_bwd_sampled = F.grid_sample(
        flow_bwd, pos_fwd_norm,
        mode='bilinear', padding_mode='zeros', align_corners=True
    )  # (B, 2, H, W)

    # Consistency error: should be close to zero
    fb_error = torch.norm(flow_fwd + flow_bwd_sampled, dim=1)  # (B, H, W)

    # Create boolean mask: True where consistent
    consistency_mask = fb_error < threshold  # (B, H, W)

    return consistency_mask, fb_error


def load_image(imfile, new_width=518, new_height=294, device='cuda'):
    img = Image.open(imfile).convert('RGB')
    img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)

    img_np = np.array(img).astype(np.uint8)  # (H, W, 3)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()  # (3, H, W)

    return img_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)

def save_vis(flow, imfile):
    flow = flow.permute(1,2,0).cpu().numpy()
    flo = flow_viz.flow_to_image(flow)
    flo = cv2.cvtColor(flo, cv2.COLOR_BGR2RGB)
    cv2.imwrite(imfile, flo)

def process_frame_raft(frame, scene_dir, model, padder, intrinsic_gts, image_prev=None, depth_prev=None, cam2world_prev=None, H_target=294, W_target=518):
    image_this_ls = []
    depth_this_ls = []
    cam2world_this_ls = []
    for cam in range(6):
        img_path = os.path.join(scene_dir, str(cam))
        image_this = load_image(os.path.join(img_path, f"{frame:0>2d}.jpg"), W_target, H_target)
        depth_this = torch.tensor(np.load(os.path.join(scene_dir, f"vggt_{cam}/{frame:0>2d}.npy")), device=DEVICE)
        cam2world_this = torch.tensor(np.loadtxt(os.path.join(img_path, f"{frame:0>2d}.txt")).astype(np.float32), device=DEVICE)  # (4, 4)
        image_this_ls.append(image_this)
        depth_this_ls.append(depth_this)
        cam2world_this_ls.append(cam2world_this)
    image_this = torch.cat(image_this_ls, dim=0)
    image_this = padder.pad(image_this)[0]
    depth_this = torch.stack(depth_this_ls, dim=0)
    cam2world_this = torch.stack(cam2world_this_ls, dim=0)
    
    if image_prev is not None:
        _, flow_fwd = model(image_prev, image_this, iters=20, test_mode=True)
        flow_fwd = padder.unpad(flow_fwd) # (6, 2, H, W)
        flow_ego_fwd = compute_ego_flow_batch(depth_prev, intrinsic_gts, cam2world_prev, cam2world_this)

        flow_dyn_fwd = flow_fwd - flow_ego_fwd
        # # flow_mag_fwd = torch.sum(flow_dyn_fwd.abs(), dim=1)
        flow_mag_fwd = torch.norm(flow_dyn_fwd, dim=1)  # (6, H, W)

        for cam in range(6):
            save_path = os.path.join(scene_dir, f'raft_{cam}')
            # save_vis(flow_fwd[cam], os.path.join(save_path, f'{frame:0>2d}_1opt.png'))
            # save_vis(flow_ego_fwd[cam], os.path.join(save_path, f'{frame:0>2d}_2ego.png'))
            # save_vis(flow_dyn_fwd[cam], os.path.join(save_path, f'{frame:0>2d}_3dyn.png'))
            # cv2.imwrite(os.path.join(save_path, f'{frame:0>2d}_4mag.png'), ((flow_mag_fwd[cam] / flow_mag_fwd[cam].max()) * 65535).to(torch.uint16).cpu().numpy())
            
            instance_path = os.path.join(scene_dir, f'openseed_{cam}/{frame-1:0>2d}_instance.png')
            if os.path.exists(instance_path):
                instance = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED)
                instance = cv2.resize(instance, (W_target, H_target), interpolation=cv2.INTER_NEAREST)
                instance = torch.from_numpy(instance).to(DEVICE)  # (H, W)
                flow_mag = torch.zeros_like(flow_mag_fwd[cam])
                for i in range(1, instance.max()+1):
                    instance_mask = instance == i
                    if instance_mask.sum() > 0:
                        flow_mag[instance_mask] = flow_mag_fwd[cam][instance_mask].mean()
                        
                flow_threshold = torch.quantile(flow_mag_fwd[cam], 0.5).item()
                flow_mask = flow_mag >= flow_threshold
                flow_mask = flow_mask.cpu().numpy().astype(np.uint8) # (H, W)
                flow_mask = cv2.dilate(flow_mask, np.ones((7, 7), np.uint8), iterations=1, dst=flow_mask)
                cv2.imwrite(os.path.join(save_path, f'{frame:0>2d}_5mask.png'), flow_mask * 255)

    image_prev = image_this
    depth_prev = depth_this
    cam2world_prev = cam2world_this
    return image_prev, depth_prev, cam2world_prev

def load_args(model_path):
    args = argparse.Namespace()
    args.model = model_path
    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False
    
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    H_target, W_target = 294, 518
    padder = InputPadder((1, 3, H_target, W_target))

    return model, padder

def main():
    data_path = sys.argv[1]
    model, padder = load_args("raft-things.pth")
    H_orig, W_orig = 900, 1600 
    H_target, W_target = 294, 518
    # H_target, W_target = 448, 768
    scale_x = W_target / W_orig
    scale_y = H_target / H_orig
    with torch.no_grad():
        for scene in sorted([scene for scene in os.listdir(data_path) if scene.startswith("scene")]): 
            scene_dir = os.path.join(data_path, scene)
            for cam in range(6):
                save_path = os.path.join(scene_dir, f'raft_{cam}')
                shutil.rmtree(save_path, ignore_errors=True)
                os.makedirs(save_path, exist_ok=True)
            intrinsic_gts = [ 
                f"{scene_dir}/0/intrinsic.txt",
                f"{scene_dir}/1/intrinsic.txt", 
                f"{scene_dir}/2/intrinsic.txt",
                f"{scene_dir}/3/intrinsic.txt",
                f"{scene_dir}/4/intrinsic.txt",
                f"{scene_dir}/5/intrinsic.txt",
            ]
            intrinsic_gts = [np.loadtxt(intrinsic_name) for intrinsic_name in intrinsic_gts]  # (3, 3)
            intrinsic_gts = np.stack(intrinsic_gts, axis=0).astype(np.float32)  # (6, 3, 3)
            intrinsic_gts = torch.from_numpy(intrinsic_gts).to(DEVICE)  # (6, 3, 3)
            intrinsic_gts[:, 0, 0] *= scale_x  # fx
            intrinsic_gts[:, 1, 1] *= scale_y  # fy
            intrinsic_gts[:, 0, 2] *= scale_x  # cx
            intrinsic_gts[:, 1, 2] *= scale_y  # cy
            start_time = time.time()
            max_frame = int((len(os.listdir(os.path.join(data_path, scene, '0')))-1)/2)
            for frame in range(max_frame): 
                if frame == 0:
                    image_prev, depth_prev, cam2world_prev = process_frame_raft(frame, scene_dir, model, padder, intrinsic_gts)
                else:
                    image_prev, depth_prev, cam2world_prev = process_frame_raft(frame, scene_dir, model, padder, intrinsic_gts, image_prev, depth_prev, cam2world_prev)
                

            print(f"{scene} Processing time per frame: {(time.time() - start_time)/max_frame*1000:.2f}ms")


if __name__ == '__main__':
    main()
