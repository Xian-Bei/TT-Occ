"""Offline mIoU aggregation (thin CLI over utils.occ_eval)."""
import json
import os
import sys
from os.path import join

from arguments import semantic_list
from custom_utils import cal_iou_miou
from utils.occ_eval import eval_miou_hist_for_scene


def format_floats_as_percentage(d):
    for key, value in d.items():
        if isinstance(value, float):
            d[key] = f"{value * 100:.2f}%"
        elif isinstance(value, dict):
            format_floats_as_percentage(value)
    return d


if __name__ == "__main__":
    setting = sys.argv[1]
    model_path = sys.argv[2]
    data_path = sys.argv[3]
    gt_path = sys.argv[4]
    mapping = json.load(open(f"{data_path}/mapping.json"))

    summary = {}
    for model in sorted(os.listdir(model_path)):
        if model.startswith('selfocc'):
            continue
        model_dir = join(model_path, model)
        if not os.path.isdir(model_dir):
            continue
        summary[model] = {'N': 0}
        hist_all_scenes = 0

        for scene in sorted(os.listdir(model_dir)):
            scene_dir = join(model_dir, scene)
            if not os.path.isdir(scene_dir):
                continue
            print("evaluating", scene, 'for', model)
            hist_one_scene = eval_miou_hist_for_scene(
                scene_dir, scene, setting, mapping[scene], gt_path,
                data_path=data_path, model_name=model,
            )
            hist_all_scenes += hist_one_scene
            summary[model]['N'] += 1

            iou, miou, mious = cal_iou_miou(hist_one_scene)
            res = format_floats_as_percentage({
                'iou': iou.item(),
                'miou': miou.item(),
                'mious': {f'{semantic_list[i]}': mious[i].item() for i in range(len(mious))},
            })
            with open(join(scene_dir, 'results.json'), "w") as f:
                f.write(json.dumps(res, indent=4))

        iou, miou, mious = cal_iou_miou(hist_all_scenes)
        summary[model]['Avg_final'] = {
            'iou': iou.item(),
            'miou': miou.item(),
            'mious': {f'{semantic_list[i]}': mious[i].item() for i in range(len(mious))},
        }
        summary[model] = format_floats_as_percentage(summary[model])
        with open(join(model_dir, f'results_{summary[model]["N"]}.json'), "w") as f:
            f.write(json.dumps(summary[model], indent=4))

    with open(join(model_path, 'results.json'), "w") as f:
        f.write(json.dumps(summary, indent=4))
    print("done")
