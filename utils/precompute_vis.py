"""Visualization helpers for test-time precompute semantic/instance outputs."""

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image

_GOLDEN_RATIO_CONJUGATE = 0.618033988749895


def build_instance_colormap_vis(instance_map: np.ndarray) -> np.ndarray:
    """Map instance IDs to distinct RGB colors; background (id 0) stays black."""
    inst = np.asarray(instance_map, dtype=np.uint16)
    h, w = inst.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    max_id = int(inst.max())
    if max_id == 0:
        return vis

    palette = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for inst_id in range(1, max_id + 1):
        hue = (inst_id * _GOLDEN_RATIO_CONJUGATE) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        palette[inst_id] = (int(r * 255), int(g * 255), int(b * 255))
    return palette[inst]


def save_instance_vis_png(instance_map: np.ndarray, save_path: str) -> None:
    vis = build_instance_colormap_vis(instance_map)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(vis).save(save_path)


def save_depth_vis_png(depth_map: np.ndarray, save_path: str, max_depth: float = 65.5) -> None:
    """Colormap depth (meters) for quick inspection; invalid/zero depth stays black."""
    import cv2

    depth = np.asarray(depth_map, dtype=np.float32).copy()
    invalid = depth <= 0
    depth = np.clip(depth, 0.0, max_depth)
    norm = (depth / max_depth * 255.0).astype(np.uint8)
    norm[invalid] = 0
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored[invalid] = 0
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, colored)
