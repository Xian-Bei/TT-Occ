#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rex-Omni + SAM Integration
Combine Rex-Omni's detection capabilities with SAM's precise segmentation.

This application:
1. Uses Rex-Omni to detect objects and get bounding boxes
2. Feeds the bounding boxes to SAM as prompts
3. Generates precise masks for each detected object
"""

from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from rex_omni import RexOmniVisualize, RexOmniWrapper

import os
from pathlib import Path
import time

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

thing_dic = {
    'bicycle': 2, # 0
    'bus': 3, 
    'car': 4, 
    'construction vehicle': 5, 
    'crane': 5,
    'motorcycle': 6,
    'person': 7, 
    'trailer': 9,
    'trailer truck': 9,
    'truck': 10,
    
}   
stuff_dic = {
    'barrier': 1,
    'cone': 8,
    'road': 11,   
    'sidewalk': 13,
    'terrain': 14,
    'grass': 14,
    
    'building': 15, 
    'wall': 15,
    'tree': 16,
    'sky': 17,
    'pedestrian': 7,
    'traffic cone': 8,
}
INDEX_MAPPING = {**thing_dic, **stuff_dic}


def setup_sam_model(
    sam_checkpoint: str = None, model_type: str = "vit_h", device: str = "cuda"
):
    """
    Initialize SAM model

    Args:
        sam_checkpoint: Path to SAM checkpoint
        model_type: SAM model type (vit_h, vit_l, vit_b)
        device: Device to run on

    Returns:
        SAM predictor
    """
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        raise ImportError(
            "Please install segment-anything: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )

    if sam_checkpoint is None:
        # Default SAM checkpoint paths
        default_checkpoints = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth",
        }
        sam_checkpoint = default_checkpoints.get(model_type)
        print(f"Using default checkpoint: {sam_checkpoint}")
        print(
            f"Please Download from: https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints"
        )

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    return predictor


def setup_rex_omni(
    model_path: str = "IDEA-Research/Rex-Omni", backend: str = "transformers"
):
    """
    Initialize Rex-Omni model

    Args:
        model_path: Path to Rex-Omni model
        backend: Backend to use (transformers or vllm)

    Returns:
        Rex-Omni wrapper
    """
    print("🚀 Initializing Rex-Omni model...")

    rex_model = RexOmniWrapper(
        model_path=model_path,
        backend=backend,
        max_tokens=2048,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
    )

    print("✅ Rex-Omni model initialized!")
    return rex_model


def detect_objects_with_rex(
    rex_model: RexOmniWrapper,
    image: Image.Image,
    categories: List[str],
    task: str = "detection",
) -> Dict:
    """
    Detect objects using Rex-Omni

    Args:
        rex_model: Rex-Omni wrapper
        image: Input image
        categories: List of object categories to detect
        task: Task type (detection, referring, etc.)

    Returns:
        Detection results with bounding boxes
    """
    # print(f"🔍 Detecting objects: {', '.join(categories)}")

    results = rex_model.inference(images=image, task=task, categories=categories)

    result = results[0]
    if not result["success"]:
        raise RuntimeError(
            f"Rex-Omni inference failed: {result.get('error', 'Unknown error')}"
        )

    return result


def extract_boxes_from_predictions(predictions: Dict) -> List[np.ndarray]:
    """
    Extract bounding boxes from Rex-Omni predictions

    Args:
        predictions: Rex-Omni predictions dictionary

    Returns:
        List of bounding boxes in [x1, y1, x2, y2] format
    """
    boxes = []
    for category, detections in predictions.items():
        for detection in detections:
            if detection["type"] == "box":
                coords = detection["coords"]
                # Ensure format is [x1, y1, x2, y2]
                box = np.array([coords[0], coords[1], coords[2], coords[3]])
                boxes.append(box)

    return boxes


def generate_masks_with_sam(
    sam_predictor, image: np.ndarray, boxes: List[np.ndarray]
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Generate masks using SAM with box prompts

    Args:
        sam_predictor: SAM predictor
        image: Input image as numpy array (H, W, 3)
        boxes: List of bounding boxes

    Returns:
        Tuple of (masks, scores)
    """
    # print(f"🎨 Generating masks for {len(boxes)} detected objects...")

    # Set image for SAM
    sam_predictor.set_image(image)

    all_masks = []
    all_scores = []

    for box in boxes:
        # SAM expects boxes in [x1, y1, x2, y2] format
        masks, scores, _ = sam_predictor.predict(
            box=box, multimask_output=False  # Get single best mask
        )

        all_masks.append(masks[0])  # Take the first (best) mask
        all_scores.append(scores[0])

    # print(f"✅ Generated {len(all_masks)} masks!")
    return all_masks, all_scores


def visualize_results(
    image: Image.Image,
    predictions: Dict,
    masks: List[np.ndarray] = None,
    save_path: str = "output_visualization.jpg",
):
    """
    Visualize detection and segmentation results

    Args:
        image: Original image
        predictions: Rex-Omni predictions
        masks: SAM masks (optional)
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(1, 2 if masks is not None else 1, figsize=(20, 10))

    if masks is None:
        axes = [axes]

    # Left: Rex-Omni detection with boxes
    rex_vis = RexOmniVisualize(
        image=image,
        predictions=predictions,
        font_size=20,
        draw_width=5,
        show_labels=True,
    )
    axes[0].imshow(rex_vis)
    axes[0].set_title("Rex-Omni Detection (Bounding Boxes)", fontsize=16)
    axes[0].axis("off")

    # Right: SAM segmentation with masks
    if masks is not None:
        img_array = np.array(image)

        # Create colored mask overlay
        mask_overlay = np.zeros_like(img_array)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(masks)))

        for mask, color in zip(masks, colors):
            mask_rgb = (mask[:, :, None] * color[:3] * 255).astype(np.uint8)
            mask_overlay = np.where(mask[:, :, None], mask_rgb, mask_overlay)

        # Blend original image with mask overlay
        alpha = 0.5
        blended = (alpha * img_array + (1 - alpha) * mask_overlay).astype(np.uint8)

        # Convert to PIL Image for drawing boxes and labels
        from PIL import ImageDraw, ImageFont

        blended_pil = Image.fromarray(blended)
        draw = ImageDraw.Draw(blended_pil)

        # Try to use a truetype font, fallback to default if not available
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
            )
        except:
            font = ImageFont.load_default()

        # Draw boxes and labels for each detection
        mask_idx = 0
        for category, detections in predictions.items():
            for detection in detections:
                if detection["type"] == "box" and mask_idx < len(masks):
                    coords = detection["coords"]
                    x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]

                    # Get corresponding color
                    color = colors[mask_idx]
                    color_rgb = tuple((color[:3] * 255).astype(int).tolist())

                    # Draw bounding box
                    draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=5)

                    # Draw label background
                    label_text = category
                    bbox = draw.textbbox((x1, y1 - 25), label_text, font=font)
                    draw.rectangle(bbox, fill=color_rgb)

                    # Draw label text
                    draw.text(
                        (x1, y1 - 25), label_text, fill=(255, 255, 255), font=font
                    )

                    mask_idx += 1

        axes[1].imshow(blended_pil)
        axes[1].set_title("SAM Segmentation (Masks + Boxes + Labels)", fontsize=16)
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"💾 Visualization saved to: {save_path}")
    plt.close()


def save_individual_masks(
    image: Image.Image,
    masks: List[np.ndarray],
    predictions: Dict,
    output_dir: str = "output_masks",
):
    """
    Save individual masks as separate images

    Args:
        image: Original image
        masks: List of masks
        predictions: Rex-Omni predictions
        output_dir: Directory to save masks
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    img_array = np.array(image)

    mask_idx = 0
    for category, detections in predictions.items():
        for det_idx, detection in enumerate(detections):
            if detection["type"] == "box" and mask_idx < len(masks):
                mask = masks[mask_idx]

                # Create masked image
                masked_img = img_array.copy()
                masked_img[~mask] = 0  # Set background to black

                # Save mask
                mask_path = os.path.join(output_dir, f"{category}_{det_idx}_mask.png")
                cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

                # Save masked image
                masked_path = os.path.join(
                    output_dir, f"{category}_{det_idx}_masked.jpg"
                )
                cv2.imwrite(masked_path, cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))

                mask_idx += 1

    print(f"💾 Individual masks saved to: {output_dir}")


def rex_sam_pipeline(
    categories: List[str],
    rex_model_path: str = "IDEA-Research/Rex-Omni",
    sam_checkpoint: str = None,
    sam_model_type: str = "vit_h",
    backend: str = "transformers",
    save_individual: bool = True,
):
    """
    Complete Rex-Omni + SAM pipeline

    Args:
        image_path: Path to input image
        categories: List of categories to detect
        rex_model_path: Path to Rex-Omni model
        sam_checkpoint: Path to SAM checkpoint
        sam_model_type: SAM model type
        backend: Rex-Omni backend
        output_path: Path to save visualization
        save_individual: Whether to save individual masks
    """
    print("=" * 60)
    print("Rex-Omni + SAM Integration Pipeline")
    print("=" * 60)

    # 2. Initialize models
    print("\n🤖 Initializing models...")
    rex_model = setup_rex_omni(model_path=rex_model_path, backend=backend)
    sam_predictor = setup_sam_model(
        sam_checkpoint=sam_checkpoint, model_type=sam_model_type
    )
    
    root = "/home/uqxsun14/datasets/nuscenes_val/"
    start = time.time()
    for scene in [scene for scene in sorted(os.listdir(root)) if scene.startswith('scene')][31:]:
        scene_path = os.path.join(root, scene)
        print(f"Processing scene: {scene}")
        # if scene != "scene-0100":
        #     continue
        # start = time.time()
        for cam in range(6):
            print(f"Processing cam: {cam}")

            cam_path = os.path.join(scene_path, f"{cam}")
            OUTPUT_DIR = Path(os.path.join(scene_path, f"gsam2_{cam}"))
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            for img_name in sorted(os.listdir(cam_path)):
                # print(img_name)
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img_path = os.path.join(cam_path, img_name)

                # 1. Load image
                # print(f"\n📷 Loading image: {img_path}")
                image = Image.open(img_path).convert("RGB")
                img_array = np.array(image)
                # print(f"✅ Image size: {image.size}")


                # 3. Detect objects with Rex-Omni
                # print(f"\n🎯 Step 1: Detecting objects with Rex-Omni...")
                result = detect_objects_with_rex(rex_model, image, categories)
                predictions = result["extracted_predictions"]

                # # Count detections
                # total_detections = sum(len(dets) for dets in predictions.values())
                # print(f"✅ Found {total_detections} objects:")
                # for cat, dets in predictions.items():
                #     print(f"   - {cat}: {len(dets)} instances")

                # 4. Extract bounding boxes
                boxes = extract_boxes_from_predictions(predictions)

                if len(boxes) == 0:
                    print(scene, cam, img_name, "⚠️  No objects detected. Exiting.")
                    continue

                # 5. Generate masks with SAM
                # print(f"\n🎨 Step 2: Generating precise masks with SAM...")
                masks, scores = generate_masks_with_sam(sam_predictor, img_array, boxes)
                # import pdb
                # pdb.set_trace()

                # print(f"✅ Mask scores: {[f'{s:.3f}' for s in scores]}")

                mask_idx = 0
                sem_masks = np.zeros_like(masks[0], dtype=np.uint8)
                sem_vis = np.zeros((sem_masks.shape[0], sem_masks.shape[1], 3), dtype=np.uint8)
                
                for category, detections in predictions.items():
                    for detection in detections:
                        if detection["type"] == "box" and mask_idx < len(masks):
                            sem_masks[masks[mask_idx]] = INDEX_MAPPING[category]
                            sem_vis[masks[mask_idx]] = colors[INDEX_MAPPING[category]]
                            mask_idx += 1

                Image.fromarray(sem_vis).save(os.path.join(OUTPUT_DIR,f"{img_name.split('.')[0]}_vis.png"))
                Image.fromarray(sem_masks, mode='L').save(os.path.join(OUTPUT_DIR, f"{img_name.split('.')[0]}.png"))
    end = time.time()
    print(f"✅ Total pipeline time: {end - start:.2f} seconds")



    # # 6. Visualize results
    # print(f"\n📊 Step 3: Creating visualizations...")
    # visualize_results(image, predictions, masks, save_path=output_path)

    # print("\n" + "=" * 60)
    # print("✅ Pipeline completed successfully!")
    # print("=" * 60)

    return {
        "predictions": predictions,
        "masks": masks,
        "scores": scores,
        "boxes": boxes,
    }


def main():
    """
    Example usage of Rex-Omni + SAM integration
    """
    # Example 1: Detect and segment people
    print("\n🌟 Example 1: Detecting and segmenting people")
    rex_sam_pipeline(
        image_path="tutorials/detection_example/test_images/cafe.jpg",
        categories=["person", "cup", "laptop"],
        output_path="applications/_1_rexomni_sam/output_cafe.jpg",
    )

    # Example 2: Referring expression with segmentation
    print("\n\n🌟 Example 2: Referring expression with segmentation")
    rex_sam_pipeline(
        image_path="tutorials/detection_example/test_images/boys.jpg",
        categories=["boy holding microphone", "boy playing piano"],
        output_path="applications/_1_rexomni_sam/output_boys.jpg",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rex-Omni + SAM Integration")

    parser.add_argument(
        "--categories",
        nargs="+",
        # default=["person", "cup", "laptop"],
        default = [
            "barrier",
            "bicycle",
            "building",
            "bus",
            "car",
            "cone",
            "construction vehicle",
            "crane",
            "grass",
            "motorcycle",
            "pedestrian",
            "person",
            "road",
            "sidewalk",
            "sky",
            "terrain",
            "trailer",
            "trailer truck",
            "traffic cone",
            "tree",
            "truck",
            "wall"
        ],
        help="Categories to detect",
    )
    parser.add_argument(
        "--rex-model",
        type=str,
        default="IDEA-Research/Rex-Omni",
        help="Rex-Omni model path",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default="sam_vit_h_4b8939.pth",
        help="SAM checkpoint path",
    )
    parser.add_argument(
        "--sam-model",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model type",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="Rex-Omni backend",
    )

    parser.add_argument(
        "--no-individual",
        action="store_true",
        default=True,
        help="Don't save individual masks",
    )

    args = parser.parse_args()

    rex_sam_pipeline(
        categories=args.categories,
        rex_model_path=args.rex_model,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model,
        backend=args.backend,
        save_individual=not args.no_individual,
    )
