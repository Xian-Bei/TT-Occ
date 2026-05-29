"""Offline nuCraft RayIoU (thin CLI over utils.occ_eval)."""
import json
import os
import sys
from os.path import join

from utils.occ_eval import eval_rayiou_nucraft_for_scene


if __name__ == "__main__":
    setting = sys.argv[1]
    model_path = sys.argv[2]
    data_path = sys.argv[3]
    gt_path = sys.argv[4]
    mapping = json.load(open(f"{data_path}/mapping.json"))

    for model in sorted(os.listdir(model_path)):
        model_dir = join(model_path, model)
        if not os.path.isdir(model_dir):
            continue
        print(f"\n=== RayIoU nuCraft: {model} ===")
        for scene in sorted(os.listdir(model_dir)):
            scene_dir = join(model_dir, scene)
            if not os.path.isdir(scene_dir):
                continue
            print("evaluating", scene, 'for', model)
            eval_rayiou_nucraft_for_scene(
                scene_dir,
                scene,
                data_path,
                mapping[scene],
                gt_path,
                setting=setting,
                model_name=model,
            )
