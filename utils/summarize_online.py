"""Aggregate saved online histograms without loading occupancy predictions or GT."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

if os.getenv('TTOCC_EXTRA_SITE_PACKAGES'):
    sys.path.append(os.environ['TTOCC_EXTRA_SITE_PACKAGES'])
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom_utils import cal_iou_miou
from arguments import semantic_list


def histogram_metrics(hist, keep_pred0=False):
    iou, miou, classes = cal_iou_miou(hist, keep_pred0=keep_pred0)
    return {
        'miou_percent': miou.item() * 100,
        'occupancy_iou_percent': iou.item() * 100,
        'finite_classes': int(torch.isfinite(classes).sum()),
        'per_class_iou_percent': {
            name: value if torch.isfinite(classes[i]) else None
            for i, (name, value) in enumerate(zip(semantic_list, (classes * 100).tolist()))
        },
    }


def summarize_online(predictions, scenes=None, histogram_file='hist.pth'):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for histogram aggregation')
    final = {}
    for model in sorted(Path(predictions).iterdir()):
        if not model.is_dir():
            continue
        names = scenes if scenes is not None else sorted(p.name for p in model.glob('scene-*') if p.is_dir())
        if not names:
            continue
        total = None
        frames = 0
        for scene in names:
            folder = model / scene
            result = json.loads((folder / 'result.json').read_text())
            hist = torch.load(folder / histogram_file, map_location='cuda', weights_only=True)
            total = hist if total is None else total + hist
            frames += result['frames']
        summary_name = 'hist_all_scenes.pth' if histogram_file == 'hist.pth' else 'hist_nucraft_all_scenes.pth'
        torch.save(total.cpu(), model / summary_name)
        final[model.name] = {
            'N': len(names), 'frames': frames, **histogram_metrics(total),
            'metrics_keep_pred0': histogram_metrics(total, keep_pred0=True),
        }
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    args = parser.parse_args()
    metrics = summarize_online(args.predictions)
    (args.predictions / 'metrics_full_precision.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))
