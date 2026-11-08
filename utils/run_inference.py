"""Run the shipped train.py inference path, with resumable scene manifests."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
if os.getenv('TTOCC_EXTRA_SITE_PACKAGES'):
    sys.path.append(os.environ['TTOCC_EXTRA_SITE_PACKAGES'])
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SOURCE_FILES = ['train.py', 'arguments/__init__.py', 'scene/__init__.py',
                'scene/dataset_readers.py', 'scene/gaussian_model.py', 'custom_utils.py',
                'utils/testtime_precompute.py', 'utils/inference_precision.py',
                'submodules/RAFT/main.py', 'submodules/OpenSeeD/main.py', 'submodules/Rex-Omni/rexsam_ttocc.py',
                'submodules/Rex-Omni/rex_omni/wrapper.py',
                'submodules/map-anything/ttocc.py',
                'submodules/map-anything/mapanything/models/mapanything/model.py',
                'utils/summarize_online.py', 'utils/run_inference.py', 'run_main.sh']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)

def timestamp():
    return datetime.now(timezone.utc).isoformat()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--gt-path', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--feature-root', type=Path)
    p.add_argument('--variants', nargs='+', choices=['lidar', 'camera_frame', 'camera_scene'], default=['lidar', 'camera_frame'])
    p.add_argument('--scenes', nargs='+')
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--occ-setting', choices=['Occ3D', 'nuCraft'], default='Occ3D')
    p.add_argument('--semantic-prefix', choices=['rexomni', 'openseed'], default='rexomni')
    p.add_argument('--openseed-text-is-eval', type=int, choices=[0, 1], default=1)
    p.add_argument('--openseed-semantic-output', choices=['panoptic', 'sem_seg'], default='sem_seg')
    p.add_argument('--openseed-instance-mode', choices=['panoptic', 'none'], default='panoptic')
    p.add_argument('--depth-prefix', choices=['mapanything', 'vggt'], default='mapanything')
    p.add_argument('--map-model', default='facebook/map-anything')
    p.add_argument('--map-revision', default='a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa')
    p.add_argument('--rex-model', default='IDEA-Research/Rex-Omni')
    p.add_argument('--rex-revision', default='0e5693d24657f6c0e091008dd6809bb1bd28988c')
    p.add_argument('--fusion', type=int, choices=[0, 1], default=1)
    p.add_argument('--fusion-unknown-abstention', type=int, choices=[0, 1], default=None,
                   help='Default: LiDAR on, Camera off; 0/1 overrides this for ablations')
    p.add_argument('--cache-only', action='store_true', help='Require cached semantics, depth, instances and motion masks; never load providers')
    p.add_argument('--histogram-only', action='store_true', help='Evaluate online without writing per-frame occupancy tensors')
    p.add_argument('--eval-both-datasets', action='store_true', help='Share fused geometry for Occ3D and nuCraft online evaluation')
    p.add_argument('--nucraft-gt-path', type=Path)
    p.add_argument('--verbose', type=int, default=1)
    args = p.parse_args()
    assert torch.cuda.is_available(), 'CUDA is required for this inference runner'
    args.data_root = args.data_root.expanduser().resolve(); args.gt_path = args.gt_path.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.eval_both_datasets:
        if args.occ_setting != 'Occ3D' or args.nucraft_gt_path is None:
            p.error('--eval-both-datasets requires --occ-setting Occ3D and --nucraft-gt-path')
        args.nucraft_gt_path = args.nucraft_gt_path.expanduser().resolve()
    features = (args.feature_root or args.output_root / 'features').expanduser().resolve()
    predictions = args.output_root / 'predictions'; logs = args.output_root / 'logs'
    for folder in [features, predictions, logs]: folder.mkdir(parents=True, exist_ok=True)
    scenes = args.scenes or sorted(x.name for x in args.data_root.glob('scene-*') if x.is_dir())
    if len(scenes) != len(set(scenes)) or not scenes: raise ValueError('Scene list is empty or has duplicates')
    mapping = json.loads((args.data_root / 'mapping.json').read_text())
    counts = {}
    for scene in scenes:
        root = args.data_root / scene
        ids = sorted(x.stem for x in (root / '0').glob('*.jpg'))
        if ids != [f'{i:02d}' for i in range(len(ids))]: raise ValueError(f'Non-contiguous frames: {scene}')
        n = min(len(ids), args.max_frames) if args.max_frames else len(ids)
        if n == 0 or scene not in mapping: raise ValueError(f'Invalid scene: {scene}')
        for c in range(6):
            if not (root / str(c) / 'intrinsic.txt').is_file(): raise FileNotFoundError(root / str(c) / 'intrinsic.txt')
            for f in range(n):
                for ext in ['jpg', 'txt']:
                    if not (root / str(c) / f'{f:02d}.{ext}').is_file(): raise FileNotFoundError(f'{scene}/{c}/{f:02d}.{ext}')
        if args.occ_setting == 'Occ3D':
            for f in range(n):
                gt = args.gt_path / scene / mapping[scene]['sample_token'][f'{f:02d}'] / 'labels.npz'
                if not gt.is_file(): raise FileNotFoundError(gt)
        if args.occ_setting == 'nuCraft' or args.eval_both_datasets:
            nu_gt = args.nucraft_gt_path if args.eval_both_datasets else args.gt_path
            for f in range(n):
                gt = nu_gt / (mapping[scene]['LIDAR_TOP'][f'{f:02d}'] + '.bin')
                if not gt.is_file(): raise FileNotFoundError(gt)
        counts[scene] = n
    source = {rel: sha(REPO / rel) for rel in SOURCE_FILES}
    abstention = {v: bool(args.fusion and (v == 'lidar' if args.fusion_unknown_abstention is None
                                         else args.fusion_unknown_abstention)) for v in args.variants}
    recipe = {'schema': 2, 'data_root': str(args.data_root), 'gt_path': str(args.gt_path),
              'occ_setting': args.occ_setting, 'semantic_prefix': args.semantic_prefix,
              'depth_prefix': args.depth_prefix, 'map_model': args.map_model,
              'map_revision': args.map_revision, 'rex_model': args.rex_model, 'rex_revision': args.rex_revision,
              'fusion': bool(args.fusion), 'fusion_K': 10, 'max_frames': args.max_frames,
              'fusion_unknown_abstention': abstention,
              'unmatched_instance_ttl': {v: 3 if v == 'lidar' else None for v in args.variants},
              'cache_only': args.cache_only, 'histogram_only': args.histogram_only,
              'eval_both_datasets': args.eval_both_datasets,
              'nucraft_gt_path': str(args.nucraft_gt_path) if args.eval_both_datasets else None,
              'openseed': {'text_is_eval': bool(args.openseed_text_is_eval),
                           'semantic_output': args.openseed_semantic_output,
                           'instance_mode': args.openseed_instance_mode} if args.semantic_prefix == 'openseed' else None,
              'rex_batch': os.getenv('TTOCC_REX_BATCH_INFER', '1'),
              'sam_box_batch': os.getenv('TTOCC_SAM_BOX_BATCH', '1'),
              'geometry': {'AMP': False, 'matmul_tf32': True},
              'depth': {'AMP': 'BF16', 'matmul_tf32': True, 'cudnn_benchmark': False,
                        'apply_mask': True, 'mask_edges': False, 'encoder_and_head_batch': 6},
              'raft': {'matmul_tf32': False, 'cudnn_benchmark': False},
              'source_sha256': source}
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    recipe_path = args.output_root / 'run_recipe.json'
    if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
        raise ValueError(f'Run recipe/source changed: {recipe_path}; use a fresh output root')
    write(recipe_path, recipe)
    write(args.output_root / 'manifest.json', {'status': 'running', 'scenes': scenes,
          'frames': sum(counts.values()), 'variants': args.variants, 'recipe_sha256': fingerprint,
          'python': sys.executable, 'torch': torch.__version__, 'cuda': torch.version.cuda,
          'gpu': torch.cuda.get_device_name(), 'started_at': timestamp()})
    for rel in SOURCE_FILES:
        dest = args.output_root / 'source_snapshot' / rel; dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists(): dest.write_bytes((REPO / rel).read_bytes())
    for rel in ['run_main.sh', 'utils/run_inference.py']:
        dest = args.output_root / 'source_snapshot' / rel; dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists(): dest.write_bytes((REPO / rel).read_bytes())
    shared = features / 'semantic'
    for variant in args.variants:
        source_kind = 'lidar' if variant == 'lidar' else 'depth'
        scope = 'scene' if variant == 'camera_scene' else 'frame'
        if scope == 'scene' and args.depth_prefix != 'mapanything': raise ValueError('Scene scope requires MapAnything')
        K = 10
        model_name = f'LIDAR_{args.semantic_prefix}' if source_kind == 'lidar' else f'Camera_{args.semantic_prefix}_{args.depth_prefix}_{scope}_RAFT'
        feature_root = shared if source_kind == 'lidar' else features / f'{args.depth_prefix}_{scope}'
        for scene in scenes:
            if any(sha(REPO / rel) != digest for rel, digest in source.items()): raise RuntimeError('Source changed during inference')
            for c in range(6):
                sem = shared / scene / f'{args.semantic_prefix}_{c}'; sem.mkdir(parents=True, exist_ok=True)
                if source_kind == 'depth':
                    link = feature_root / scene / sem.name; link.parent.mkdir(parents=True, exist_ok=True)
                    if not link.exists(): link.symlink_to(sem, target_is_directory=True)
            dest = predictions / model_name / scene; done = dest / 'completed.json'
            signature = {'recipe_sha256': fingerprint, 'variant': variant, 'frames': counts[scene], 'K': K}
            if done.exists():
                if json.loads(done.read_text()) != signature: raise ValueError(f'Completed scene recipe mismatch: {done}')
                print('SKIP_COMPLETED', variant, scene, flush=True); continue
            dest.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, '-B', str(REPO / 'train.py'), '-m', str(dest), '-s', str(args.data_root / scene),
                   '--feature_root', str(feature_root), '--occ_setting', args.occ_setting,
                   '--variant', source_kind, '--K', str(K), '--max_frames', str(args.max_frames),
                   '--eval_occ', '--occ3d_path' if args.occ_setting == 'Occ3D' else '--nucraft_path', str(args.gt_path),
                   '--enable_testtime_precompute', '--precompute_prefetch',
                   '--precompute_use_disk', '1', '--precompute_write_disk', '1', '--precompute_save_vis', '0',
                   '--semantic_prefix', args.semantic_prefix, '--depth_prefix', args.depth_prefix,
                   '--openseed_text_is_eval', str(args.openseed_text_is_eval),
                   '--openseed_semantic_output', args.openseed_semantic_output,
                   '--openseed_instance_mode', args.openseed_instance_mode,
                   '--mapanything_scope', scope, '--mapanything_model_path', args.map_model,
                   '--mapanything_revision', args.map_revision, '--rexomni_model_path', args.rex_model,
                   '--rexomni_revision', args.rex_revision,
                   '--mapanything_apply_mask', '1', '--mapanything_mask_edges', '0', '--verbose', str(args.verbose)]
            if args.fusion: cmd.append('--use_fusion')
            cmd += ['--fusion_unknown_abstention', str(int(abstention[variant]))]
            if args.cache_only:
                cmd.remove('--enable_testtime_precompute')
                cmd.remove('--precompute_prefetch')
                cmd += ['--precompute_cache_only']
            if args.histogram_only: cmd.append('--histogram_only')
            if args.eval_both_datasets:
                cmd += ['--eval_both_datasets', '--nucraft_path', str(args.nucraft_gt_path)]
            cmd.append('--disable_viewer')
            write(args.output_root / 'progress.json', {'status': 'running', 'variant': variant, 'scene': scene,
                  'scene_index': scenes.index(scene)+1, 'scene_count': len(scenes), 'command': cmd, 'started_at': timestamp()})
            ts = time.monotonic(); print('RUN', variant, scene, flush=True)
            with (logs / f'{variant}_{scene}.log').open('a') as stream:
                subprocess.run(cmd, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
            result = json.loads((dest / 'result.json').read_text())
            expected = {f'{i:02d}.pth' for i in range(counts[scene])}
            actual = {x.name for x in (dest / 'Occ').glob('*.pth')}
            if (result['frames'] != counts[scene] or
                (not args.histogram_only and actual != expected) or not (dest / 'hist.pth').is_file() or
                (args.eval_both_datasets and not (dest / 'hist_nucraft.pth').is_file())):
                raise RuntimeError(f'Incomplete inference output: {dest}')
            write(done, signature)
            print('COMPLETED_SCENE', variant, scene, 'seconds', round(time.monotonic()-ts, 1), flush=True)
    from utils.summarize_online import summarize_online
    final = summarize_online(predictions, scenes=scenes)
    if args.eval_both_datasets:
        final = {'Occ3D': final, 'nuCraft': summarize_online(
            predictions, scenes=scenes, histogram_file='hist_nucraft.pth')}
    write(args.output_root / 'metrics_full_precision.json', final)
    meta = json.loads((args.output_root / 'manifest.json').read_text()); meta.update(status='completed', completed_at=timestamp(), metrics=final)
    write(args.output_root / 'manifest.json', meta); write(args.output_root / 'progress.json', {'status': 'completed', 'metrics': final})
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Keep a durable failure state for unattended runs; never count a failed
        # or incomplete scene as completed.
        if '--output-root' in sys.argv:
            output = Path(sys.argv[sys.argv.index('--output-root')+1]).expanduser()
            failure = {'status': 'failed', 'failed_at': timestamp(), 'error': str(exc),
                       'traceback': traceback.format_exc()}
            write(output / 'failure.json', failure)
            progress_path = output / 'progress.json'
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
            progress.update(failure); write(progress_path, progress)
            manifest_path = output / 'manifest.json'
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                manifest.update(status='failed', failed_at=failure['failed_at'], error=str(exc))
                write(manifest_path, manifest)
        raise
