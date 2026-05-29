# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Demo script to get MapAnything outputs in COLMAP format. Optionally can also run BA on outputs.

Reference: VGGT (https://github.com/facebookresearch/vggt/blob/main/demo_colmap.py)
"""

import argparse
import copy
import glob
import os
import shutil
from torchvision import transforms as TF
import cv2

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from torchvision import transforms as tvf

from mapanything.models import MapAnything
from mapanything.third_party.np_to_pycolmap import (
    batch_np_matrix_to_pycolmap,
    batch_np_matrix_to_pycolmap_wo_track,
)
from mapanything.third_party.track_predict import predict_tracks
from mapanything.utils.geometry import closed_form_pose_inverse, depthmap_to_world_frame
from mapanything.utils.image import rgb
from mapanything.utils.misc import seed_everything
from mapanything.utils.viz import predictions_to_glb
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

import gc

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def parse_args():
    parser = argparse.ArgumentParser(description="MapAnything COLMAP Demo")
    parser.add_argument(
        "--scene_dir",
        type=str,
        default="",
        required=False,
        help="Directory containing the scene images",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--memory_efficient_inference",
        action="store_true",
        default=False,
        help="Use memory efficient inference for reconstruction (trades off speed)",
    )
    parser.add_argument(
        "--conf_thres_value",
        type=float,
        default=0.0,
        help="Confidence threshold value for depth filtering (used only without BA)",
    )
    parser.add_argument(
        "--save_glb",
        action="store_true",
        default=False,
        help="Save dense reconstruction (without BA) as GLB file",
    )
    parser.add_argument(
        "--use_ba", action="store_true", default=False, help="Use BA for reconstruction"
    )
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error",
        type=float,
        default=8.0,
        help="Maximum reprojection error for reconstruction",
    )
    parser.add_argument(
        "--shared_camera",
        action="store_true",
        default=False,
        help="Use shared camera for all images",
    )
    parser.add_argument(
        "--camera_type",
        type=str,
        default="SIMPLE_PINHOLE",
        help="Camera type for reconstruction",
    )
    parser.add_argument(
        "--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks"
    )
    parser.add_argument(
        "--query_frame_num", type=int, default=8, help="Number of frames to query"
    )
    parser.add_argument(
        "--max_query_pts", type=int, default=4096, help="Maximum number of query points"
    )
    parser.add_argument(
        "--fine_tracking",
        action="store_true",
        default=True,
        help="Use fine tracking (slower but more accurate)",
    )
    return parser.parse_args()


def load_and_preprocess_images_square(
    image_path_list, target_size=1024, data_norm_type=None
):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 1024.
        data_norm_type (str, optional): Image normalization type. See UniCeption IMAGE_NORMALIZATION_DICT keys. Defaults to None (no normalization).

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 5) containing [x1, y1, x2, y2, width, height] for each image
        )

    Raises:
        ValueError: If the input list is empty or if an invalid data_norm_type is provided
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []  # Renamed from position_info to be more descriptive

    # Set up normalization based on data_norm_type
    if data_norm_type is None:
        # No normalization, just convert to tensor
        img_transform = tvf.ToTensor()
    elif data_norm_type in IMAGE_NORMALIZATION_DICT.keys():
        # Use the specified normalization
        img_norm = IMAGE_NORMALIZATION_DICT[data_norm_type]
        img_transform = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
        )
    else:
        raise ValueError(
            f"Unknown image normalization type: {data_norm_type}. Available options: {list(IMAGE_NORMALIZATION_DICT.keys())}"
        )

    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        # Convert to RGB
        img = img.convert("RGB")

        # Get original dimensions
        width, height = img.size

        # Make the image square by padding the shorter dimension
        max_dim = max(width, height)

        # Calculate padding
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # Calculate scale factor for resizing
        scale = target_size / max_dim

        # Calculate final coordinates of original image in target space
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        # Store original image coordinates and scale
        original_coords.append(np.array([x1, y1, x2, y2, width, height]))

        # Create a new black square image and paste original
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # Resize to target size
        square_img = square_img.resize(
            (target_size, target_size), Image.Resampling.BICUBIC
        )

        # Convert to tensor and apply normalization
        img_tensor = img_transform(square_img)
        images.append(img_tensor)

    # Stack all images
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)

    return images, original_coords


def randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    """
    If mask has more than max_trues True values,
    randomly keep only max_trues of them and set the rest to False.
    """
    # 1D positions of all True entries
    true_indices = np.flatnonzero(mask)  # shape = (N_true,)

    # if already within budget, return as-is
    if true_indices.size <= max_trues:
        return mask

    # randomly pick which True positions to keep
    sampled_indices = np.random.choice(
        true_indices, size=max_trues, replace=False
    )  # shape = (max_trues,)

    # build new flat mask: True only at sampled positions
    limited_flat_mask = np.zeros(mask.size, dtype=bool)
    limited_flat_mask[sampled_indices] = True

    # restore original shape
    return limited_flat_mask.reshape(mask.shape)


def create_pixel_coordinate_grid(num_frames, height, width):
    """
    Creates a grid of pixel coordinates and frame indices for all frames.
    Returns:
        tuple: A tuple containing:
            - points_xyf (numpy.ndarray): Array of shape (num_frames, height, width, 3)
                                            with x, y coordinates and frame indices
            - y_coords (numpy.ndarray): Array of y coordinates for all frames
            - x_coords (numpy.ndarray): Array of x coordinates for all frames
            - f_coords (numpy.ndarray): Array of frame indices for all frames
    """
    # Create coordinate grids for a single frame
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    # Broadcast to all frames
    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    # Create frame indices and broadcast
    f_idx = np.arange(num_frames, dtype=np.float32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    # Stack coordinates and frame indices
    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

    return points_xyf


def run_mapanything(
    model,
    images,
    poses,
    intrinsics,
    dtype,
    image_normalization_type="dinov2",
    memory_efficient_inference=False,
):
    # Images: [V, 3, H, W]
    # Check image shape
    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # Hard-coded to use 518 for MapAnything
    # images = F.interpolate(
    #     images, size=(resolution, resolution), mode="bilinear", align_corners=False
    # )

    # Run inference
    views = []
    num_cams = 6
    assert images.shape[0] % num_cams == 0, "Number of images must be multiple of number of cameras"
    # import pdb; pdb.set_trace()
    for view_idx in range(images.shape[0]):
        cam_idx = view_idx % num_cams
        view = {
            "img": images[view_idx][None],  # Add batch dimension
            "data_norm_type": [image_normalization_type],
            "intrinsics": intrinsics[cam_idx][None],
            "camera_poses": poses[view_idx][None],  # Add batch dimension
        }
        views.append(view)
    del images, poses, intrinsics
    predictions = model.infer(
        views, memory_efficient_inference=memory_efficient_inference
    )

    # Process predictions
    (
        all_extrinsics,
        all_intrinsics,
        all_depth_maps,
        all_depth_confs,
        all_pts3d,
        all_img_no_norm,
        all_masks,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for pred in predictions:
        # Compute 3D points from depth, intrinsics, and camera pose
        depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
        intrinsics_torch = pred["intrinsics"][0]  # (3, 3)
        camera_pose_torch = pred["camera_poses"][0]  # (4, 4)
        pts3d, valid_mask = depthmap_to_world_frame(
            depthmap_torch, intrinsics_torch, camera_pose_torch
        )

        # Extract mask from predictions and combine with valid depth mask
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = mask & valid_mask.cpu().numpy()  # Combine with valid depth mask

        # Convert tensors to numpy arrays
        extrinsic = (
            closed_form_pose_inverse(pred["camera_poses"])[0].cpu().numpy()
        )  # c2w -> w2c
        intrinsic = intrinsics_torch.cpu().numpy()
        depth_map = depthmap_torch.cpu().numpy()
        depth_conf = pred["conf"][0].cpu().numpy()
        pts3d = pts3d.cpu().numpy()
        img_no_norm = pred["img_no_norm"][0].cpu().numpy()  # Denormalized image

        # Collect results
        all_extrinsics.append(extrinsic)
        all_intrinsics.append(intrinsic)
        all_depth_maps.append(depth_map)
        all_depth_confs.append(depth_conf)
        all_pts3d.append(pts3d)
        all_img_no_norm.append(img_no_norm)
        all_masks.append(mask)

    # Stack results into arrays
    all_extrinsics = np.stack(all_extrinsics)
    all_intrinsics = np.stack(all_intrinsics)
    all_depth_maps = np.stack(all_depth_maps)
    all_depth_confs = np.stack(all_depth_confs)
    all_pts3d = np.stack(all_pts3d)
    all_img_no_norm = np.stack(all_img_no_norm)
    all_masks = np.stack(all_masks)

    return (
        all_extrinsics,
        all_intrinsics,
        all_depth_maps,
        all_depth_confs,
        all_pts3d,
        all_img_no_norm,
        all_masks,
    )

def load_and_preprocess_images(image_path_list, mode="crop", semantic=False):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    
    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:

        # Open image
        img = Image.open(image_path)

        if not semantic:
            # If there's an alpha channel, blend onto white background:
            if img.mode == "RGBA":
                # Create white background
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                # Alpha composite onto the white background
                img = Image.alpha_composite(background, img)

            # Now convert to "RGB" (this step assigns white for transparent areas)
            img = img.convert("RGB")

        width, height = img.size
        
        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        if not semantic:
            img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            img = to_tensor(img)  # Convert to tensor (0, 1)
        else:
            img = img.resize((new_width, new_height), Image.Resampling.NEAREST)
            img = torch.from_numpy(np.array(img))

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]
        
        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]
            
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                
                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[-2], img.shape[-1]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images

# def demo_fn(args):
#     # Print configuration
#     print("Arguments:", vars(args))

#     # Set seed for reproducibility
#     seed_everything(args.seed)

#     # Set device and dtype
#     dtype = (
#         torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
#     )
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
#     print(f"Using dtype: {dtype}")

#     # Init model
#     print("Loading MapAnything model from huggingface ...")
#     model = MapAnything.from_pretrained("facebook/map-anything").to(device)
#     model.eval()

#     data_root="/home/uqxsun14/datasets/nuscenes_val"
#     H_orig, W_orig = 900, 1600 
#     H_target, W_target = 294, 518
#     scale_x = W_target / W_orig
#     scale_y = H_target / H_orig
#     for scene in sorted([scene for scene in os.listdir(data_root) if scene.startswith("scene")]): 
#         print(f"Processing {scene}")
#         scene_dir = os.path.join(data_root, scene)
#         new_world2old_world = np.loadtxt(os.path.join(scene_dir, "0/00.txt"))
#         old_world2new_world = np.linalg.inv(new_world2old_world)
#         intrinsic3x3_ls = []
#         for cam in range(6):
#             save_dir = os.path.join(scene_dir, f"mapanything3_{cam}")
#             shutil.rmtree(save_dir, ignore_errors=True)
#             os.makedirs(save_dir, exist_ok=True)
#             intrinsic3x3 = np.loadtxt(os.path.join(scene_dir, f"{cam}/intrinsic.txt"))
#             intrinsic3x3 = torch.tensor(intrinsic3x3, dtype=torch.float32)
#             intrinsic3x3[0, 0] *= scale_x  # fx
#             intrinsic3x3[1, 1] *= scale_y  # fy
#             intrinsic3x3[0, 2] *= scale_x  # cx
#             intrinsic3x3[1, 2] *= scale_y  # cy
#             # import pdb; pdb.set_trace()
#             intrinsic3x3_ls.append(intrinsic3x3)
#         max_frame = int((len(os.listdir(os.path.join(scene_dir, "0")))-1)/2)
#         for frame in range(0, max_frame):
#             device = "cuda"
#             image_names = [ f"{scene_dir}/0/{frame:0>2d}.jpg", 
#                             f"{scene_dir}/1/{frame:0>2d}.jpg", 
#                             f"{scene_dir}/2/{frame:0>2d}.jpg",
#                             f"{scene_dir}/3/{frame:0>2d}.jpg",
#                             f"{scene_dir}/4/{frame:0>2d}.jpg",
#                             f"{scene_dir}/5/{frame:0>2d}.jpg",]
#             pose_names = [ f"{scene_dir}/0/{frame:0>2d}.txt", 
#                             f"{scene_dir}/1/{frame:0>2d}.txt", 
#                             f"{scene_dir}/2/{frame:0>2d}.txt",
#                             f"{scene_dir}/3/{frame:0>2d}.txt",
#                             f"{scene_dir}/4/{frame:0>2d}.txt",
#                             f"{scene_dir}/5/{frame:0>2d}.txt",]
#             images_this = load_and_preprocess_images(image_names).to(device)
#             images = images_this.to(device)
#             world2cam_list = []
#             cam2world_list = []
#             for pose_path in pose_names:
#                 cam2world = np.loadtxt(pose_path)
#                 cam2world = old_world2new_world @ cam2world
#                 # world2cam = np.linalg.inv(cam2world)
#                 cam2world = torch.tensor(cam2world, dtype=torch.float32, device=device)
#                 # world2cam = torch.tensor(world2cam, dtype=torch.float32, device=device)
#                 cam2world_list.append(cam2world)
#                 # world2cam_list.append(world2cam)
#             extrinsic, intrinsic, depth_map, depth_conf, points_3d, img_no_norm, masks = (
#                 run_mapanything(
#                     model,
#                     images,
#                     cam2world_list,
#                     intrinsic3x3_ls,
#                     dtype,
#                     model.encoder.data_norm_type,
#                     memory_efficient_inference=args.memory_efficient_inference,
#                 )
#             )
#             depth_vggt = depth_map
#             # import pdb; pdb.set_trace()
#             for cam in range(6):
#                 save_dir = os.path.join(scene_dir, f"mapanything3_{cam}")
#                 depth = depth_vggt[cam]
#                 # print(depth.shape)
#                 np.save(os.path.join(save_dir, f'{frame:0>2d}'), depth)
#                 depth[depth > 65.5] = 65.5
#                 depth_img = (depth * 1000).astype(np.uint16)
#                 cv2.imwrite(os.path.join(save_dir, f'{frame:0>2d}.png'), depth_img)
#             # import pdb; pdb.set_trace()

#     return True

def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    # Set seed for reproducibility
    seed_everything(args.seed)

    # Set device and dtype
    dtype = (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Init model
    print("Loading MapAnything model from huggingface ...")
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    model.eval()

    data_root="/home/uqxsun14/datasets/nuscenes_val"
    H_orig, W_orig = 900, 1600 
    H_target, W_target = 294, 518
    scale_x = W_target / W_orig
    scale_y = H_target / H_orig

    for scene in sorted([scene for scene in os.listdir(data_root) if scene.startswith("scene")]): 
        print(f"Processing {scene}")
        intrinsic3x3_ls_all = []
        image_names_all = []
        pose_names_all = []
        scene_dir = os.path.join(data_root, scene)
        new_world2old_world = np.loadtxt(os.path.join(scene_dir, "0/00.txt"))
        old_world2new_world = np.linalg.inv(new_world2old_world)
        for cam in range(6):
            save_dir = os.path.join(scene_dir, f"mapanything3_{cam}")
            shutil.rmtree(save_dir, ignore_errors=True)
            os.makedirs(save_dir, exist_ok=True)
            intrinsic3x3 = np.loadtxt(os.path.join(scene_dir, f"{cam}/intrinsic.txt"))
            intrinsic3x3 = torch.tensor(intrinsic3x3, dtype=torch.float32)
            intrinsic3x3[0, 0] *= scale_x  # fx
            intrinsic3x3[1, 1] *= scale_y  # fy
            intrinsic3x3[0, 2] *= scale_x  # cx
            intrinsic3x3[1, 2] *= scale_y  # cy
            # import pdb; pdb.set_trace()
            intrinsic3x3_ls_all.append(intrinsic3x3)
        max_frame = int((len(os.listdir(os.path.join(scene_dir, "0")))-1)/2)
        for frame in range(0, max_frame):
            device = "cuda"
            image_names = [ f"{scene_dir}/0/{frame:0>2d}.jpg", 
                            f"{scene_dir}/1/{frame:0>2d}.jpg", 
                            f"{scene_dir}/2/{frame:0>2d}.jpg",
                            f"{scene_dir}/3/{frame:0>2d}.jpg",
                            f"{scene_dir}/4/{frame:0>2d}.jpg",
                            f"{scene_dir}/5/{frame:0>2d}.jpg",]
            pose_names = [  f"{scene_dir}/0/{frame:0>2d}.txt", 
                            f"{scene_dir}/1/{frame:0>2d}.txt", 
                            f"{scene_dir}/2/{frame:0>2d}.txt",
                            f"{scene_dir}/3/{frame:0>2d}.txt",
                            f"{scene_dir}/4/{frame:0>2d}.txt",
                            f"{scene_dir}/5/{frame:0>2d}.txt",]
            image_names_all.extend(image_names)
            pose_names_all.extend(pose_names)
        images_this = load_and_preprocess_images(image_names_all).to(device)
        images_this = images_this.to(device)
        cam2world_list = []
        for pose_path in pose_names_all:
            cam2world = np.loadtxt(pose_path)
            cam2world = old_world2new_world @ cam2world
            # world2cam = np.linalg.inv(cam2world)
            cam2world = torch.tensor(cam2world, dtype=torch.float32, device=device)
            # world2cam = torch.tensor(world2cam, dtype=torch.float32, device=device)
            cam2world_list.append(cam2world)
            # world2cam_list.append(world2cam)
        _, _, depth_map, _, _, _, _ = (
            run_mapanything(
                model,
                images_this,
                cam2world_list,
                intrinsic3x3_ls_all,
                dtype,
                model.encoder.data_norm_type,
                memory_efficient_inference=args.memory_efficient_inference,
            )
        )
        # import pdb; pdb.set_trace()
        num_cams = 6
        for frame in range(max_frame):
            for cam in range(num_cams):
                idx = frame * num_cams + cam
                save_dir = os.path.join(scene_dir, f"mapanything3_{cam}")
                os.makedirs(save_dir, exist_ok=True)
                depth = depth_map[idx]
                # print(depth.shape)
                np.save(os.path.join(save_dir, f'{frame:0>2d}'), depth)
                depth[depth > 65.5] = 65.5
                depth_img = (depth * 1000).astype(np.uint16)
                cv2.imwrite(os.path.join(save_dir, f'{frame:0>2d}.png'), depth_img)
            # import pdb; pdb.set_trace()
        del images_this, cam2world_list, depth_map
        torch.cuda.empty_cache()         # 释放未使用显存
        torch.cuda.ipc_collect()         # 清理内部共享缓存
        gc.collect()                     # 清理 Python 引用
        torch.cuda.synchronize()         # 保证上一次 GPU 操作完全结束
        print(f"[GPU cleared after {scene}]")
    return True

def rename_colmap_recons_and_rescale_camera(
    reconstruction,
    image_paths,
    original_coords,
    img_size,
    shift_point2d_to_original_res=False,
    shared_camera=False,
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded & resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # No need to rescale any more
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)
