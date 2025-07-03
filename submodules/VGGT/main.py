import sys
import numpy as np
import cv2
import shutil
import ipdb
import os
import time
from os.path import join
import open3d as o3d
import torch
torch.set_printoptions(sci_mode=False)
import torch.nn.functional as F
from vggt.utils.visual_track import visualize_tracks_on_images
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def create_camera_frustum(K, cam2world, image_size, scale=0.2):
    H, W = image_size
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    z = scale
    corners = np.array([
        [(0 - cx) * z / fx, (0 - cy) * z / fy, z],
        [(W - cx) * z / fx, (0 - cy) * z / fy, z],
        [(W - cx) * z / fx, (H - cy) * z / fy, z],
        [(0 - cx) * z / fx, (H - cy) * z / fy, z],
    ])
    origin = np.array([[0, 0, 0]])
    frustum_cam = np.vstack([origin, corners])

    frustum_cam_h = np.hstack([frustum_cam, np.ones((5, 1))])
    frustum_world = (cam2world @ frustum_cam_h.T).T[:, :3]

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]
    colors = [[1, 0, 0] for _ in lines]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(frustum_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def visualize_scale_alignment_with_frustum(pts3d, pred_pts3d, K, image_size):
    geometries = []

    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(cam_frame)

    frustum = create_camera_frustum(K, np.eye(4), image_size, scale=2.0)
    geometries.append(frustum)

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(pts3d)
    pcd_gt.paint_uniform_color([0.1, 0.7, 0.2])
    geometries.append(pcd_gt)

    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(pred_pts3d)
    pcd_pred.paint_uniform_color([0.7, 0.2, 0.2])
    geometries.append(pcd_pred)

    o3d.visualization.draw_geometries(geometries)

def generate_sparse_pixel_coords(H_target, W_target, stride=1, device='cpu'):
    y_coords  = torch.arange(0, H_target, stride, device=device) # shape (H_s,)
    x_coords1 = torch.arange(0, W_target//5, stride, device=device)
    x_coords2 = torch.arange(W_target-1, W_target-1-W_target//5, -stride, device=device)
    x_coords = torch.cat([x_coords1, x_coords2], dim=0)  # shape (W_s,)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords = torch.stack([xx, yy], dim=-1)  # (H_s, W_s, 2)

    coords = coords.view(-1, 2)
    return coords

def triangulate_with_pixel_indices(
    track, valid_mask, intrinsic_gts, world2cam_gts
):
    """
        track: (3, N, 2)
        vis_score: (3, N)
        conf_score: (3, N)
        intrinsic_gts: (3, 3, 3)
        world2cam_gts: (3, 4, 4)

        pts3d_all: (M, 3)
        idxs_all: (M,) idx in track[0]
    """
    pts3d_list = []
    idxs_list = []
    P0 = intrinsic_gts[0] @ world2cam_gts[0][:3, :]

    mask01 = valid_mask[0] & valid_mask[1]
    idx01 = np.where(mask01)[0]
    if len(idx01) > 0:
        pts0 = track[0][idx01].T
        pts1 = track[1][idx01].T
        P1 = intrinsic_gts[1] @ world2cam_gts[1][:3, :]
        pts4d = cv2.triangulatePoints(P0, P1, pts0, pts1)
        pts3d = (pts4d[:3] / pts4d[3]).T
        pts3d_list.append(pts3d)
        idxs_list.append(idx01)

    mask02 = valid_mask[0] & valid_mask[2]
    idx02 = np.where(mask02)[0]
    if len(idx02) > 0:
        pts0 = track[0][idx02].T
        pts2 = track[2][idx02].T
        P2 = intrinsic_gts[2] @ world2cam_gts[2][:3, :]
        pts4d = cv2.triangulatePoints(P0, P2, pts0, pts2)
        pts3d = (pts4d[:3] / pts4d[3]).T
        pts3d_list.append(pts3d)
        idxs_list.append(idx02)

    if len(pts3d_list) == 0:
        return np.empty((0, 3)), np.empty((0,), dtype=int)

    pts3d_all = np.concatenate(pts3d_list, axis=0)
    idxs_all = np.concatenate(idxs_list, axis=0)
    
    cam0_pose = world2cam_gts[0]  # (4, 4)
    R = cam0_pose[:3, :3]
    t = cam0_pose[:3, 3]

    pts_cam0 = (R @ pts3d_all.T + t[:, None]).T  # (N, 3)
    z_valid = pts_cam0[:, 2] > 0
    dist_valid = np.linalg.norm(pts_cam0, axis=1) < 30.0
    mask_valid = z_valid & dist_valid

    pts3d_all = pts3d_all[mask_valid]
    pts_cam0 = pts_cam0[mask_valid]
    idxs_all = idxs_all[mask_valid]

    return pts_cam0, idxs_all

def estimate_scale_from_depth_alignment(pts3d, pixel_coords, depth_map, K):
    """
        pts3d: (M, 3) 
        pixel_coords: (M, 2) 
        depth_map: (H, W) 
        K: (3, 3) 
    """
    H, W = depth_map.shape
    u = np.round(pixel_coords[:, 0]).astype(int)
    v = np.round(pixel_coords[:, 1]).astype(int)

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u[valid], v[valid]
    pts3d_valid = pts3d[valid]
    pixel_coords_valid = pixel_coords[valid]

    depth_pred = depth_map[v, u]

    x = (pixel_coords_valid[:, 0] - K[0, 2]) * depth_pred / K[0, 0]
    y = (pixel_coords_valid[:, 1] - K[1, 2]) * depth_pred / K[1, 1]
    z = depth_pred
    pred_pts3d = np.stack([x, y, z], axis=1)

    norm_gt = np.linalg.norm(pts3d_valid, axis=1)
    norm_pred = np.linalg.norm(pred_pts3d, axis=1)
    valid_mask = norm_pred > 1e-6

    if valid_mask.sum() == 0:
        return 0.0

    scale = np.median(norm_gt[valid_mask] / norm_pred[valid_mask])
    return scale.item()


def process_frame_vggt(frame, scene_dir, intrinsic_gts, prev_scale, model, depth_time=0, scale_time=0, H_target=294, W_target=518):
    device = "cuda"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    # Initialize the model and load the pretrained weights.
    threshold = 0.2
    cam2world_gts = [   
                    f"{scene_dir}/0/{frame:0>2d}.txt", 
                    f"{scene_dir}/1/{frame:0>2d}.txt", 
                    f"{scene_dir}/2/{frame:0>2d}.txt",
                    f"{scene_dir}/3/{frame:0>2d}.txt",
                    f"{scene_dir}/4/{frame:0>2d}.txt",
                    f"{scene_dir}/5/{frame:0>2d}.txt",
                ]
    cam2world_gts = np.stack([np.loadtxt(cam2world_gt) for cam2world_gt in cam2world_gts], axis=0).astype(np.float32) # (6, 4, 4)
    cam2world_gts_cuda = torch.from_numpy(cam2world_gts).to(device)  # (6, 4, 4)
    world2cam_gts = np.linalg.inv(cam2world_gts)  # (6, 4, 4)
    with torch.cuda.amp.autocast(dtype=dtype):
        start_time = time.time()
        image_names = [ f"{scene_dir}/0/{frame:0>2d}.jpg", 
                        f"{scene_dir}/1/{frame:0>2d}.jpg", 
                        f"{scene_dir}/2/{frame:0>2d}.jpg",
                        f"{scene_dir}/3/{frame:0>2d}.jpg",
                        f"{scene_dir}/4/{frame:0>2d}.jpg",
                        f"{scene_dir}/5/{frame:0>2d}.jpg",]  
        images_this = load_and_preprocess_images(image_names).to(device)[None] # (1, 6, 3, 294, 518)
        ###############################################
        aggregated_tokens_list, ps_idx = model.aggregator(images_this)
        depth_vggt_cuda, depth_conf_org = model.depth_head(aggregated_tokens_list, images_this, ps_idx) # (B, S, H, W, 1) (B, S, H, W)
        depth_time += time.time() - start_time
        
        depth_vggt_cuda = depth_vggt_cuda.squeeze()  # (S, H, W)
        depth_vggt = depth_vggt_cuda.cpu().numpy()  # (S, H, W

        # Predict Tracks for Triangulation
        start_time = time.time()
        images = images_this[:,[0,1,5]] # (1, 3, 3, 294, 518)
        aggregated_tokens_list, ps_idx = model.aggregator(images)
        query_points = generate_sparse_pixel_coords(H_target, W_target, stride=10, device=device).unsqueeze(0) # (N, 2)
        track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points)
    track = track_list[-1]
    track = track.squeeze().cpu().numpy() # (3, N, 2)
    world2cam = world2cam_gts[[0,1,5]] # (3, 4, 4)
    intrinsic = intrinsic_gts[[0,1,5]] # (3, 3, 3)
    valid_mask = ((vis_score > threshold) & (conf_score > threshold)).squeeze().cpu().numpy()  # (3, N)

    pts3d, idxs = triangulate_with_pixel_indices(
        track=track,
        valid_mask=valid_mask,
        world2cam_gts=world2cam,
        intrinsic_gts=intrinsic,
    )
    scale = estimate_scale_from_depth_alignment(
        pts3d=pts3d,  
        pixel_coords=track[0][idxs],
        depth_map=depth_vggt[0],
        K=intrinsic_gts[0]
    )
    scale_time += time.time() - start_time
    if scale == 0:
        scale = prev_scale
    # print(f"Scale: {scale:.2f}")
    visualize_tracks_on_images(f'{frame:02d}', images, track_list[-1], (conf_score>0.2) & (vis_score>0.2), out_dir=os.path.join(scene_dir, "vggt_track"))
    

    # pixel_coords = track[0][idxs]  # (N, 2)
    # u = pixel_coords[:, 0]
    # v = pixel_coords[:, 1]
    # depth = depth_vggt[0][v.astype(int), u.astype(int)]
    # x = (u - intrinsic_gts[0][0, 2]) * depth / intrinsic_gts[0][0, 0]
    # y = (v - intrinsic_gts[0][1, 2]) * depth / intrinsic_gts[0][1, 1]
    # z = depth
    # pred_pts = np.stack([x, y, z], axis=1)
    # pred_pts_scaled = pred_pts * scale
    # visualize_scale_alignment_with_frustum(
    #     pts3d=pts3d,
    #     pred_pts3d=pred_pts_scaled,
    #     K=intrinsic_gts[0],
    #     image_size=(H_target, W_target)
    # )

    depth_vggt *= scale
    depth_vggt_cuda *= scale
    for cam in range(6):
        save_dir = os.path.join(scene_dir, f"vggt_{cam}")
        depth = depth_vggt[cam]
        np.save(os.path.join(save_dir, f'{frame:0>2d}'), depth)
        depth[depth > 65.5] = 65.5
        depth_img = (depth * 1000).astype(np.uint16)
        cv2.imwrite(os.path.join(save_dir, f'{frame:0>2d}.png'), depth_img)
    
    prev_scale = scale
    return prev_scale, depth_time, scale_time
    
    
def main():
    device = "cuda"
    root = sys.argv[1] 
    # This will automatically download the model weights the first time it's run, which may take a while.
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    with torch.no_grad():
        H_orig, W_orig = 900, 1600 
        H_target, W_target = 294, 518
        scale_x = W_target / W_orig
        scale_y = H_target / H_orig
        for scene in sorted([scene for scene in os.listdir(root) if scene.startswith("scene")]): 
            depth_time = 0
            scale_time = 0
            dynamic_time = 0
            print(f"Processing {scene}")
            scene_dir = os.path.join(root, scene)
            for cam in range(6):
                save_dir = os.path.join(scene_dir, f"vggt_{cam}")
                shutil.rmtree(save_dir, ignore_errors=True)
                os.makedirs(save_dir, exist_ok=True)
            max_frame = int((len(os.listdir(os.path.join(scene_dir, "0")))-1)/2)
            os.makedirs(os.path.join(scene_dir, "vggt_track"), exist_ok=True)

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
            intrinsic_gts[:, 0, 0] *= scale_x  # fx
            intrinsic_gts[:, 1, 1] *= scale_y  # fy
            intrinsic_gts[:, 0, 2] *= scale_x  # cx
            intrinsic_gts[:, 1, 2] *= scale_y  # cy
            intrinsic_gts_cuda = torch.from_numpy(intrinsic_gts).to(device)  # (6, 3, 3)

            prev_scale = 20
            for frame in range(0, max_frame):
                prev_scale, depth_time, scale_time = process_frame_vggt(frame, scene_dir, intrinsic_gts, prev_scale, model, depth_time, scale_time)
                
                    
            print(f"Finished {scene}, depth time: {depth_time/max_frame*1000:.0f} ms, scale time: {scale_time/max_frame*1000:.0f} ms")
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()