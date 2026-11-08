import os
import sys
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from utils.inference_precision import precision_stage
from PIL import Image


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _validate_openseed_cache(scene_feature_dir, opt):
    if str(opt.semantic_prefix).lower() != "openseed":
        return
    import json
    expected = {"text_is_eval": bool(getattr(opt, "openseed_text_is_eval", 0)),
                "semantic_output": getattr(opt, "openseed_semantic_output", "panoptic"),
                "instance_mode": getattr(opt, "openseed_instance_mode", "panoptic")}
    semantic_dir = Path(scene_feature_dir, f"{opt.semantic_prefix}_0")
    recipe = semantic_dir.resolve().parent / "openseed_recipe.json"
    if recipe.exists():
        if json.loads(recipe.read_text()) != expected:
            raise ValueError(f"OpenSeeD cache settings mismatch: {recipe}")
    elif (expected != {"text_is_eval": False, "semantic_output": "panoptic", "instance_mode": "panoptic"}
          and any(semantic_dir.glob("*.png"))):
        raise ValueError(f"Untagged OpenSeeD cache cannot establish requested settings: {semantic_dir}")


def precomputed_scene_dir(scene_path, feature_root=""):
    if feature_root:
        return os.path.join(os.path.abspath(feature_root), os.path.basename(os.path.normpath(scene_path)))
    return scene_path

def scene_feature_requirements(
    source="lidar",
    semantic_prefix="openseed",
    depth_prefix="vggt",
    depth_ext="npy",
    dynamic_mask_prefix="raft",
    dynamic_mask_suffix="_5mask.png",
):
    reqs = [f"{semantic_prefix}_{cam}" for cam in range(6)]
    if source == "depth":
        reqs += [f"{depth_prefix}_{cam}" for cam in range(6)]
    return reqs


def _lazy_import_openseed():
    import importlib.util
    submodule_root = _repo_root() / "submodules" / "OpenSeeD"
    if str(submodule_root) not in sys.path:
        sys.path.insert(0, str(submodule_root))
    # OpenSeeD's build_ext --inplace leaves its CUDA extension in this folder.
    ops_root = str(submodule_root / "openseed" / "body" / "encoder" / "ops")
    if ops_root not in sys.path:
        sys.path.append(ops_root)
    # Merge OpenSeeD utils into already-loaded namespace package "utils".
    # Without this, imports like "from utils.constants import *" may still resolve
    # only to the repo-level utils path and miss OpenSeeD's utils modules.
    if "utils" in sys.modules:
        openseed_utils = str(submodule_root / "utils")
        utils_mod = sys.modules["utils"]
        utils_path = getattr(utils_mod, "__path__", None)
        if utils_path is not None and openseed_utils not in list(utils_path):
            utils_path.append(openseed_utils)
    mod_path = submodule_root / "main.py"
    spec = importlib.util.spec_from_file_location("ttocc_openseed_main", mod_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lazy_import_vggt():
    import importlib.util
    submodule_root = _repo_root() / "submodules" / "VGGT"
    if str(submodule_root) not in sys.path:
        sys.path.insert(0, str(submodule_root))
    mod_path = submodule_root / "main.py"
    spec = importlib.util.spec_from_file_location("ttocc_vggt_main", mod_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lazy_import_raft():
    import importlib.util
    submodule_root = _repo_root() / "submodules" / "RAFT"
    if str(submodule_root) not in sys.path:
        sys.path.insert(0, str(submodule_root))
    mod_path = submodule_root / "main.py"
    spec = importlib.util.spec_from_file_location("ttocc_raft_main", mod_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lazy_import_rexomni():
    import importlib.util
    submodule_root = _repo_root() / "submodules" / "Rex-Omni"
    if str(submodule_root) not in sys.path:
        sys.path.insert(0, str(submodule_root))
    mod_path = submodule_root / "rexsam_ttocc.py"
    spec = importlib.util.spec_from_file_location("ttocc_rexomni_main", mod_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lazy_import_mapanything():
    import importlib.util
    submodule_root = _repo_root() / "submodules" / "map-anything"
    if str(submodule_root) not in sys.path:
        sys.path.insert(0, str(submodule_root))
    mod_path = submodule_root / "ttocc.py"
    spec = importlib.util.spec_from_file_location("ttocc_mapanything_main", mod_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _touch(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(str(time.time()))


def _should_materialize_feature_dirs(opt):
    return bool(opt.precompute_write_disk or opt.precompute_save_vis)


def _ensure_semantic_output_dirs(scene_feature_dir, opt):
    for cam in range(6):
        os.makedirs(os.path.join(scene_feature_dir, f"{opt.semantic_prefix}_{cam}"), exist_ok=True)


def _ensure_raft_vis_dirs(scene_feature_dir, opt):
    for cam in range(6):
        os.makedirs(
            os.path.join(scene_feature_dir, f"{opt.dynamic_mask_prefix}_{cam}", "vis"),
            exist_ok=True,
        )


def _ensure_depth_vis_dirs(scene_feature_dir, opt):
    for cam in range(6):
        os.makedirs(
            os.path.join(scene_feature_dir, f"{opt.depth_prefix}_{cam}", "vis"),
            exist_ok=True,
        )


def _save_depth_frame_vis(depths, scene_feature_dir, frame_id, depth_prefix):
    from utils.precompute_vis import save_depth_vis_png

    frame_name = f"{int(frame_id):0>2d}"
    for cam, depth in depths.items():
        vis_path = os.path.join(
            scene_feature_dir,
            f"{depth_prefix}_{cam}",
            "vis",
            f"{frame_name}_depth_vis.png",
        )
        save_depth_vis_png(depth, vis_path)


def _scene_lock_file(scene_name, scene_feature_dir, opt):
    if _should_materialize_feature_dirs(opt):
        return os.path.join(scene_feature_dir, ".lock")
    lock_root = Path(tempfile.gettempdir()) / "ttocc_precompute_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    return str(lock_root / f"{scene_name}.lock")


class OpenSeeDRunner:
    @precision_stage(matmul_tf32=True)
    def __init__(self, semantic_prefix="openseed", use_inference_mode=True, text_is_eval=False,
                 semantic_output="panoptic", instance_mode="panoptic"):
        mod = _lazy_import_openseed()
        self.semantic_prefix = semantic_prefix
        self.semantic_output = semantic_output
        self.instance_mode = instance_mode
        self.runtime = mod.build_openseed_runtime(
            conf_file="submodules/OpenSeeD/configs/openseed/openseed_swint_lang.yaml",
            weight_path="model_state_dict_swint_51.2ap.pt",
            use_inference_mode=use_inference_mode,
            text_is_eval=text_is_eval,
        )
        self._mod = mod

    @precision_stage(matmul_tf32=True)
    def process_scene(self, scene_path, scene_feature_dir, save_vis=False):
        if (self.runtime["text_is_eval"] or self.semantic_output != "panoptic"
                or self.instance_mode != "panoptic"):
            for path in sorted(Path(scene_path, "0").glob("*.jpg")):
                self.process_frame(scene_path, scene_feature_dir, int(path.stem),
                                   save_vis=save_vis, write_disk=True)
            return {"peak_mb": int(torch.cuda.max_memory_reserved() / 1024**2)}
        return self._mod.process_scene(
            self.runtime,
            scene_in=scene_path,
            scene_out=scene_feature_dir,
            save_vis=save_vis,
        )

    @precision_stage(matmul_tf32=True)
    def process_frame(self, scene_path, scene_feature_dir, frame_id, save_vis=False, write_disk=True,
                      semantic_output=None, instance_mode=None):
        return self._mod.process_frame_openseed(
            self.runtime,
            scene_in=scene_path,
            scene_out=scene_feature_dir,
            frame=int(frame_id),
            save_vis=save_vis,
            write_disk=write_disk,
            semantic_prefix=self.semantic_prefix,
            semantic_output=semantic_output or self.semantic_output,
            instance_mode=instance_mode or self.instance_mode,
        )


class VGGTRunner:
    @precision_stage(matmul_tf32=True)
    def __init__(self, use_inference_mode=True):
        mod = _lazy_import_vggt()
        self.runtime = mod.build_vggt_runtime(device="cuda")
        self.use_inference_mode = use_inference_mode
        self._mod = mod

    @precision_stage(matmul_tf32=True)
    def process_scene(self, scene_path, scene_feature_dir, frame_step=1):
        max_frame = int((len(os.listdir(os.path.join(scene_path, "0"))) - 1) / 2)
        return self._mod.process_scene_vggt(
            self.runtime,
            scene_in=scene_path,
            scene_out=scene_feature_dir,
            start_frame=0,
            end_frame=max_frame,
            write_png=False,
            use_inference_mode=self.use_inference_mode,
        )

    @precision_stage(matmul_tf32=True)
    def process_frame(
        self,
        scene_path,
        scene_feature_dir,
        frame_id,
        prev_scale,
        write_disk=True,
        save_vis=False,
        depth_prefix="vggt",
    ):
        outputs = self._mod.process_frame_vggt_api(
            self.runtime,
            scene_in=scene_path,
            scene_out=scene_feature_dir,
            frame=int(frame_id),
            prev_scale=prev_scale,
            use_inference_mode=self.use_inference_mode,
            write_disk=write_disk,
            write_png=False,
        )
        if save_vis:
            _save_depth_frame_vis(outputs["depths"], scene_feature_dir, frame_id, depth_prefix)
        return outputs


class RAFTRunner:
    @precision_stage(matmul_tf32=False)
    def __init__(self, use_inference_mode=True):
        mod = _lazy_import_raft()
        self.runtime = mod.build_raft_runtime(model_path="submodules/RAFT/raft-things.pth")
        self.use_inference_mode = use_inference_mode
        self._mod = mod

    @precision_stage(matmul_tf32=False)
    def process_scene(self, scene_path, scene_feature_dir, frame_step=1):
        max_frame = int((len(os.listdir(os.path.join(scene_path, "0"))) - 1) / 2)
        return self._mod.process_scene_raft(
            self.runtime,
            scene_in=scene_path,
            scene_out=scene_feature_dir,
            start_frame=0,
            end_frame=max_frame,
            save_debug_vis=False,
            use_inference_mode=self.use_inference_mode,
        )

    @precision_stage(matmul_tf32=False)
    def process_frame(
        self,
        scene_path,
        scene_feature_dir,
        frame_id,
        state,
        depth_overrides=None,
        prev_instance_overrides=None,
        save_vis=False,
        write_disk=False,
        dynamic_mask_prefix="raft",
        dynamic_mask_suffix="_5mask.png",
    ):
        return self._mod.process_frame_raft_api(
            self.runtime,
            scene_in=scene_path,
            scene_out=scene_feature_dir,
            frame=int(frame_id),
            state=state,
            use_inference_mode=self.use_inference_mode,
            write_disk=write_disk,
            save_vis=save_vis,
            dynamic_mask_prefix=dynamic_mask_prefix,
            dynamic_mask_suffix=dynamic_mask_suffix,
            depth_overrides=depth_overrides,
            prev_instance_overrides=prev_instance_overrides,
        )


class RexOmniRunner:
    _DEFAULT_CATEGORIES = [
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
        "wall",
    ]

    @precision_stage(matmul_tf32=True)
    def __init__(
        self,
        semantic_prefix="rexomni",
        model_path="IDEA-Research/Rex-Omni",
        model_revision="0e5693d24657f6c0e091008dd6809bb1bd28988c",
        backend="transformers",
        sam_checkpoint="submodules/Rex-Omni/checkpoints/sam_vit_h_4b8939.pth",
        sam_model_type="vit_h",
        use_inference_mode=True,
    ):
        mod = _lazy_import_rexomni()
        self._mod = mod
        self.semantic_prefix = semantic_prefix
        self.categories = list(self._DEFAULT_CATEGORIES)
        self.use_inference_mode = use_inference_mode
        self.rex_batch_infer = _env_flag("TTOCC_REX_BATCH_INFER", default=True)
        self.sam_box_batch = _env_flag("TTOCC_SAM_BOX_BATCH", default=False)
        checkpoint_path = sam_checkpoint
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = str(_repo_root() / checkpoint_path)
        if not os.path.isdir(model_path):
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(model_path, revision=model_revision)
        self.rex_model = mod.setup_rex_omni(model_path=model_path, backend=backend)
        self.sam_predictor = mod.setup_sam_model(
            sam_checkpoint=checkpoint_path,
            model_type=sam_model_type,
        )

    def _build_semantic_instance(self, img_array, predictions):
        h, w = img_array.shape[:2]
        semantic = np.zeros((h, w), dtype=np.uint8)
        instance = np.zeros((h, w), dtype=np.uint8)
        boxes = self._mod.extract_boxes_from_predictions(predictions)
        if len(boxes) == 0:
            return semantic, instance

        if self.sam_box_batch:
            masks, _ = self._mod.generate_masks_with_sam_box_batch(self.sam_predictor, img_array, boxes)
        else:
            masks, _ = self._mod.generate_masks_with_sam(self.sam_predictor, img_array, boxes)
        mask_idx = 0
        next_instance_id = 1
        for category, detections in predictions.items():
            for detection in detections:
                if detection["type"] != "box" or mask_idx >= len(masks):
                    continue
                mask = masks[mask_idx]
                sem_id = self._mod.INDEX_MAPPING.get(category)
                if sem_id is not None:
                    semantic[mask] = sem_id
                if sem_id in self._mod.THING_INSTANCE_IDS:
                    instance[mask] = min(next_instance_id, 255)
                    next_instance_id += 1
                mask_idx += 1
        return semantic, instance

    @staticmethod
    def _load_rgb(image_path):
        image = Image.open(image_path).convert("RGB")
        return image, np.array(image)

    @precision_stage(matmul_tf32=True)
    def process_frame(self, scene_path, scene_feature_dir, frame_id, save_vis=False, write_disk=True):
        frame_name = f"{int(frame_id):0>2d}"
        image_paths = [os.path.join(scene_path, f"{cam}", f"{frame_name}.jpg") for cam in range(6)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            loaded = list(pool.map(self._load_rgb, image_paths))
        images = [item[0] for item in loaded]
        img_arrays = [item[1] for item in loaded]
        inference_ctx = torch.inference_mode if self.use_inference_mode else torch.no_grad
        if (
            self.rex_batch_infer
            and hasattr(self._mod, "detect_objects_with_rex_batch")
        ):
            with inference_ctx():
                batched_results = self._mod.detect_objects_with_rex_batch(
                    self.rex_model,
                    images,
                    self.categories,
                )
        else:
            batched_results = []
            with inference_ctx():
                for image in images:
                    batched_results.append(
                        self._mod.detect_objects_with_rex(
                            self.rex_model,
                            image,
                            self.categories,
                        )
                    )

        frame_outputs = {}
        for cam in range(6):
            predictions = batched_results[cam]["extracted_predictions"]
            semantic, instance = self._build_semantic_instance(img_arrays[cam], predictions)
            frame_outputs[cam] = {"semantic": semantic, "instance": instance}
            out_dir = os.path.join(scene_feature_dir, f"{self.semantic_prefix}_{cam}")
            if write_disk:
                os.makedirs(out_dir, exist_ok=True)
                Image.fromarray(semantic).save(os.path.join(out_dir, f"{frame_name}.png"))
                Image.fromarray(instance).save(os.path.join(out_dir, f"{frame_name}_instance.png"))
            if save_vis:
                vis_dir = os.path.join(out_dir, "vis")
                self._mod.save_semantic_vis_png(
                    semantic,
                    os.path.join(vis_dir, f"{frame_name}_vis.png"),
                )
                self._mod.save_instance_vis_png(
                    instance,
                    os.path.join(vis_dir, f"{frame_name}_instance_vis.png"),
                )
        return frame_outputs


class MapAnythingRunner:
    @precision_stage(matmul_tf32=True)
    def __init__(self, depth_prefix="mapanything", model_path="facebook/map-anything", use_inference_mode=True,
                 revision="a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa", scope="frame", apply_mask=True, mask_edges=False):
        mod = _lazy_import_mapanything()
        self._mod = mod
        self.depth_prefix = depth_prefix
        self.use_inference_mode = use_inference_mode
        self.model = mod.MapAnything.from_pretrained(model_path, revision=revision, strict=True).to("cuda")
        self.model.inference_minibatch_size = 6
        self.model.eval()
        self.device = "cuda"
        self._scene_static = {}
        self._pose_cache = {}
        self.memory_efficient_inference = True
        if scope not in {"frame", "scene"}:
            raise ValueError(f"Unsupported mapanything_scope: {scope}")
        self.scope, self.apply_mask, self.mask_edges = scope, apply_mask, mask_edges
        self.recipe = {"model": model_path, "revision": revision, "scope": scope,
                       "apply_mask": apply_mask, "mask_edges": mask_edges,
                       "encoder_batch": 6, "head_batch": 6, "amp": "bf16",
                       "matmul_tf32": True, "cudnn_benchmark": False,
                       "encoder_registers": bool(getattr(self.model, "use_register_tokens_from_encoder", False))}
        self._whole_scene_key = None
        self._whole_scene_depths = None

    def _scene_meta(self, scene_path):
        scene_name = os.path.basename(os.path.normpath(scene_path))
        cached = self._scene_static.get(scene_name)
        if cached is not None:
            return cached
        new_world2old_world = np.loadtxt(os.path.join(scene_path, "0", "00.txt"))
        old_world2new_world = np.linalg.inv(new_world2old_world)
        h_orig, w_orig = 900, 1600
        h_target, w_target = 294, 518
        scale_x = w_target / w_orig
        scale_y = h_target / h_orig
        intrinsics = []
        for cam in range(6):
            intrinsic3x3 = np.loadtxt(os.path.join(scene_path, f"{cam}", "intrinsic.txt"))
            intrinsic3x3 = torch.tensor(intrinsic3x3, dtype=torch.float32)
            intrinsic3x3[0, 0] *= scale_x
            intrinsic3x3[1, 1] *= scale_y
            intrinsic3x3[0, 2] *= scale_x
            intrinsic3x3[1, 2] *= scale_y
            intrinsics.append(intrinsic3x3)
        cached = {
            "old_world2new_world": old_world2new_world,
            "intrinsics": intrinsics,
        }
        self._scene_static[scene_name] = cached
        return cached

    def _load_pose(self, pose_path):
        return np.loadtxt(pose_path)

    def _load_frame_poses(self, scene_path, frame_name, old_world2new_world):
        cache_key = (scene_path, frame_name)
        cached = self._pose_cache.get(cache_key)
        if cached is not None:
            return cached
        pose_names = [os.path.join(scene_path, f"{cam}", f"{frame_name}.txt") for cam in range(6)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            raw_poses = list(pool.map(self._load_pose, pose_names))
        poses = []
        for pose in raw_poses:
            cam2world = old_world2new_world @ pose
            cam2world = torch.tensor(cam2world, dtype=torch.float32, device=self.device)
            poses.append(cam2world)
        self._pose_cache[cache_key] = poses
        return poses

    @precision_stage(matmul_tf32=True)
    def process_frame(self, scene_path, scene_feature_dir, frame_id, write_disk=True, save_vis=False):
        frame_id = int(frame_id)
        frame_name = f"{frame_id:0>2d}"
        meta = self._scene_meta(scene_path)
        image_names = [os.path.join(scene_path, f"{cam}", f"{frame_name}.jpg") for cam in range(6)]
        self._check_recipe(scene_feature_dir, write_disk)
        if self.scope == "scene":
            if self._whole_scene_key != os.path.abspath(scene_path):
                frames = len(list(Path(scene_path, "0").glob("*.jpg")))
                image_names = [os.path.join(scene_path, str(c), f"{f:02d}.jpg")
                               for f in range(frames) for c in range(6)]
                images = self._mod.load_and_preprocess_images(image_names).to(self.device)
                poses = [p for f in range(frames) for p in self._load_frame_poses(
                    scene_path, f"{f:02d}", meta["old_world2new_world"])]
                all_depth = self._infer(images, poses, meta["intrinsics"])
                self._whole_scene_depths = all_depth.reshape(frames, 6, *all_depth.shape[-2:])
                self._whole_scene_key = os.path.abspath(scene_path)
                if write_disk:
                    for f in range(frames):
                        for c in range(6):
                            folder = Path(scene_feature_dir, f"{self.depth_prefix}_{c}")
                            folder.mkdir(parents=True, exist_ok=True)
                            np.save(folder / f"{f:02d}.npy", self._whole_scene_depths[f, c])
            depth_map = self._whole_scene_depths[frame_id]
        else:
            images = self._mod.load_and_preprocess_images(image_names).to(self.device)
            poses = self._load_frame_poses(scene_path, frame_name, meta["old_world2new_world"])
            depth_map = self._infer(images, poses, meta["intrinsics"])

        depths = {}
        for cam in range(6):
            depth = depth_map[cam].astype(np.float32)
            depths[cam] = depth
            if write_disk:
                out_dir = os.path.join(scene_feature_dir, f"{self.depth_prefix}_{cam}")
                os.makedirs(out_dir, exist_ok=True)
                np.save(os.path.join(out_dir, frame_name), depth)
        if save_vis:
            _save_depth_frame_vis(depths, scene_feature_dir, frame_id, self.depth_prefix)
        return {"depths": depths}

    def _check_recipe(self, scene_feature_dir, write_disk):
        import json
        path = Path(scene_feature_dir, "mapanything_recipe.json")
        if path.exists() and json.loads(path.read_text()) != self.recipe:
            raise ValueError(f"Depth cache recipe mismatch: {path}; choose a fresh feature_root")
        if write_disk and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.recipe, indent=2) + "\n")

    def _infer(self, images, poses, intrinsics):
        inference_ctx = torch.inference_mode if self.use_inference_mode else torch.no_grad
        with inference_ctx():
            return self._mod.run_mapanything_depth_only(
                self.model, images, poses, intrinsics, self.model.encoder.data_norm_type,
                memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                apply_mask=self.apply_mask, mask_edges=self.mask_edges)


class ScenePrecomputeManager:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._scene_locks = {}
        self._scene_locks_guard = threading.Lock()
        self._prefetch_executor = ThreadPoolExecutor(max_workers=2)
        self._prefetch_futures = {}
        self._frame_prefetch_futures = {}
        self._openseed_runner = None
        self._rexomni_runner = None
        self._vggt_runner = None
        self._mapanything_runner = None
        self._raft_runner = None
        self._runner_lock = threading.Lock()
        self._memory_cache = {}
        self._cache_order = {}
        self._cache_keep_frames = 3
        self._vggt_state = {}
        self._raft_state = {}
        self._raft_state_frame = {}
        self._frame_profiles = {}

    @classmethod
    def get(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def _get_scene_lock(self, scene_name):
        with self._scene_locks_guard:
            if scene_name not in self._scene_locks:
                self._scene_locks[scene_name] = threading.Lock()
            return self._scene_locks[scene_name]

    def _cache_scene(self, scene_name):
        if scene_name not in self._memory_cache:
            self._memory_cache[scene_name] = {}
            self._cache_order[scene_name] = []
        return self._memory_cache[scene_name]

    def _put_frame_cache(self, scene_name, frame_id, frame_data):
        scene_cache = self._cache_scene(scene_name)
        frame_key = int(frame_id)
        scene_cache[frame_key] = frame_data
        order = self._cache_order[scene_name]
        if frame_key in order:
            order.remove(frame_key)
        order.append(frame_key)
        while len(order) > self._cache_keep_frames:
            evict_key = order.pop(0)
            scene_cache.pop(evict_key, None)

    def get_cached_frame_features(self, scene_path, frame_id):
        scene_name = os.path.basename(os.path.normpath(scene_path))
        scene_cache = self._memory_cache.get(scene_name, {})
        return scene_cache.get(int(frame_id))

    def get_frame_profile(self, scene_path, frame_id):
        scene_name = os.path.basename(os.path.normpath(scene_path))
        frame_key = int(frame_id)
        profile = self._frame_profiles.get((scene_name, frame_key))
        if profile is None:
            return {
                "status": "unknown",
                "total_s": 0.0,
                "semantic_s": 0.0,
                "depth_s": 0.0,
                "raft_s": 0.0,
                "raft_warmup_s": 0.0,
            }
        return profile

    def _with_repo_cwd(self):
        cwd = os.getcwd()
        os.chdir(str(_repo_root()))
        return cwd

    def _ensure_semantic_runner(self, opt):
        with self._runner_lock:
            prev_cwd = self._with_repo_cwd()
            try:
                semantic_key = str(opt.semantic_prefix).lower()
                if semantic_key == "openseed":
                    if self._openseed_runner is None:
                        self._openseed_runner = OpenSeeDRunner(
                            semantic_prefix=opt.semantic_prefix,
                            use_inference_mode=opt.precompute_inference_mode,
                            text_is_eval=bool(getattr(opt, "openseed_text_is_eval", 0)),
                            semantic_output=getattr(opt, "openseed_semantic_output", "panoptic"),
                            instance_mode=getattr(opt, "openseed_instance_mode", "panoptic"),
                        )
                elif semantic_key == "rexomni":
                    if self._rexomni_runner is None:
                        self._rexomni_runner = RexOmniRunner(
                            semantic_prefix=opt.semantic_prefix,
                            use_inference_mode=opt.precompute_inference_mode,
                            model_path=opt.rexomni_model_path,
                            model_revision=opt.rexomni_revision,
                        )
                else:
                    raise ValueError(f"Unsupported semantic_prefix: {opt.semantic_prefix}")
            finally:
                os.chdir(prev_cwd)

    def _ensure_depth_runner(self, opt):
        if opt.variant != "depth":
            return
        with self._runner_lock:
            prev_cwd = self._with_repo_cwd()
            try:
                depth_key = str(opt.depth_prefix).lower()
                if depth_key == "vggt":
                    if self._vggt_runner is None:
                        self._vggt_runner = VGGTRunner(
                            use_inference_mode=opt.precompute_inference_mode
                        )
                elif depth_key == "mapanything":
                    if self._mapanything_runner is None:
                        self._mapanything_runner = MapAnythingRunner(
                            depth_prefix=opt.depth_prefix,
                            use_inference_mode=opt.precompute_inference_mode,
                            model_path=opt.mapanything_model_path,
                            revision=opt.mapanything_revision,
                            scope=opt.mapanything_scope,
                            apply_mask=bool(opt.mapanything_apply_mask),
                            mask_edges=bool(opt.mapanything_mask_edges),
                        )
                else:
                    raise ValueError(f"Unsupported depth_prefix: {opt.depth_prefix}")
            finally:
                os.chdir(prev_cwd)

    def _ensure_raft_runner(self, opt):
        if opt.variant != "depth":
            return
        with self._runner_lock:
            prev_cwd = self._with_repo_cwd()
            try:
                if self._raft_runner is None:
                    self._raft_runner = RAFTRunner(use_inference_mode=opt.precompute_inference_mode)
            finally:
                os.chdir(prev_cwd)

    def _ensure_all_runners(self, opt):
        """Load every runner (scene-wide offline precompute only)."""
        self._ensure_semantic_runner(opt)
        self._ensure_depth_runner(opt)
        self._ensure_raft_runner(opt)

    def _semantic_runner(self, opt):
        semantic_key = str(opt.semantic_prefix).lower()
        if semantic_key == "openseed":
            return self._openseed_runner
        if semantic_key == "rexomni":
            return self._rexomni_runner
        raise ValueError(f"Unsupported semantic_prefix: {opt.semantic_prefix}")

    def _depth_runner(self, opt):
        depth_key = str(opt.depth_prefix).lower()
        if depth_key == "vggt":
            return self._vggt_runner
        if depth_key == "mapanything":
            return self._mapanything_runner
        raise ValueError(f"Unsupported depth_prefix: {opt.depth_prefix}")

    @staticmethod
    def _raft_runner_kwargs(opt):
        return {
            "save_vis": bool(opt.precompute_save_vis),
            "write_disk": bool(opt.precompute_write_disk),
            "dynamic_mask_prefix": opt.dynamic_mask_prefix,
            "dynamic_mask_suffix": opt.dynamic_mask_suffix,
        }

    @staticmethod
    def _iter_precomputed_frame_ids(scene_feature_dir, semantic_prefix):
        sem_dir = os.path.join(scene_feature_dir, f"{semantic_prefix}_0")
        if not os.path.isdir(sem_dir):
            return []
        frame_ids = []
        for name in os.listdir(sem_dir):
            if not name.endswith(".png") or name.endswith("_instance.png") or name.endswith("_vis.png"):
                continue
            stem = name[:-4]
            if len(stem) == 2 and stem.isdigit():
                frame_ids.append(int(stem))
        return sorted(frame_ids)

    def is_scene_ready(self, scene_feature_dir, source, opt, scene_path=None):
        reqs = scene_feature_requirements(
            source=source,
            semantic_prefix=opt.semantic_prefix,
            depth_prefix=opt.depth_prefix,
            depth_ext=opt.depth_ext,
            dynamic_mask_prefix=opt.dynamic_mask_prefix,
            dynamic_mask_suffix=opt.dynamic_mask_suffix,
        )
        if not os.path.exists(scene_feature_dir):
            return False
        for rel in reqs:
            p = os.path.join(scene_feature_dir, rel)
            if not os.path.exists(p):
                return False
            if os.path.isdir(p):
                if len(os.listdir(p)) == 0:
                    return False

        if scene_path is not None:
            cam0 = os.path.join(scene_path, "0")
            if not os.path.isdir(cam0):
                return False
            max_frame = int((len(os.listdir(cam0)) - 1) / 2)
            if max_frame <= 0:
                return False
            for frame_id in range(max_frame):
                if self._frame_readiness_state(scene_feature_dir, frame_id, source, opt) != "full_ready":
                    return False
            return True

        frame_ids = self._iter_precomputed_frame_ids(scene_feature_dir, opt.semantic_prefix)
        if not frame_ids:
            return False
        for frame_id in frame_ids:
            if self._frame_readiness_state(scene_feature_dir, frame_id, source, opt) != "full_ready":
                return False
        return True

    @staticmethod
    def _component_complete(frame_cache, key):
        if frame_cache is None:
            return False
        comp = frame_cache.get(key)
        if not isinstance(comp, dict):
            return False
        return all(cam in comp for cam in range(6))

    @staticmethod
    def _zero_instance_mask_like(semantic_map):
        return np.zeros(semantic_map.shape[:2], dtype=np.uint8)

    def _read_instance_cam(self, scene_feature_dir, opt, cam, frame_name, semantic_map):
        ins_path = os.path.join(
            scene_feature_dir,
            f"{opt.semantic_prefix}_{cam}",
            f"{frame_name}_instance.png",
        )
        if os.path.exists(ins_path):
            return self._read_png(ins_path)
        return self._zero_instance_mask_like(semantic_map)

    def _fill_missing_instances(self, frame_cache):
        semantic = frame_cache.get("semantic", {})
        instance = frame_cache.setdefault("instance", {})
        for cam in range(6):
            if cam in semantic and cam not in instance:
                instance[cam] = self._zero_instance_mask_like(semantic[cam])
        return frame_cache

    def _frame_readiness_state(self, scene_feature_dir, frame_id, source, opt):
        if not os.path.exists(scene_feature_dir):
            return "not_ready"
        frame_name = f"{int(frame_id):0>2d}"
        sem_ok = True
        for cam in range(6):
            sem_ok = sem_ok and os.path.exists(
                os.path.join(scene_feature_dir, f"{opt.semantic_prefix}_{cam}", f"{frame_name}.png")
            )
        depth_ok = True
        if source == "depth":
            for cam in range(6):
                depth_ok = depth_ok and os.path.exists(
                    os.path.join(scene_feature_dir, f"{opt.depth_prefix}_{cam}", f"{frame_name}.{opt.depth_ext}")
                )
        # The default online path computes RAFT. Cache-only runs must also
        # provide instance maps and the stored motion masks (except frame 0).
        if getattr(opt, "precompute_cache_only", False):
            for cam in range(6):
                sem_ok = sem_ok and os.path.isfile(os.path.join(
                    scene_feature_dir, f"{opt.semantic_prefix}_{cam}", f"{frame_name}_instance.png"))
                if source == "depth" and int(frame_id) > 0:
                    depth_ok = depth_ok and os.path.isfile(os.path.join(
                        scene_feature_dir, f"{opt.dynamic_mask_prefix}_{cam}",
                        f"{frame_name}{opt.dynamic_mask_suffix}"))
        full_ready = sem_ok and depth_ok
        if full_ready:
            return "full_ready"
        ins_ok = any(
            os.path.exists(
                os.path.join(scene_feature_dir, f"{opt.semantic_prefix}_{cam}", f"{frame_name}_instance.png")
            )
            for cam in range(6)
        )
        any_hit = sem_ok or ins_ok
        if source == "depth":
            any_hit = any_hit or depth_ok
        return "partial_ready" if any_hit else "not_ready"

    def _load_frame_cache_from_disk_partial(self, scene_feature_dir, frame_id, source, opt):
        frame_name = f"{int(frame_id):0>2d}"
        frame_cache = {"semantic": {}, "instance": {}}
        for cam in range(6):
            sem_path = os.path.join(scene_feature_dir, f"{opt.semantic_prefix}_{cam}", f"{frame_name}.png")
            if os.path.exists(sem_path):
                frame_cache["semantic"][cam] = self._read_png(sem_path)
            if cam in frame_cache["semantic"]:
                frame_cache["instance"][cam] = self._read_instance_cam(
                    scene_feature_dir, opt, cam, frame_name, frame_cache["semantic"][cam]
                )
        if source == "depth":
            frame_cache["depth"] = {}
            for cam in range(6):
                depth_path = os.path.join(scene_feature_dir, f"{opt.depth_prefix}_{cam}", f"{frame_name}.{opt.depth_ext}")
                if os.path.exists(depth_path):
                    frame_cache["depth"][cam] = self._read_depth(depth_path, opt.depth_ext)
        return frame_cache

    def _load_or_compute_frame_cache_without_raft(
        self,
        scene_path,
        scene_feature_dir,
        frame_id,
        source,
        opt,
        write_disk,
        save_vis,
    ):
        frame_id = int(frame_id)
        cached = self.get_cached_frame_features(scene_path, frame_id)
        if cached is None:
            cached = {}
        disk_partial = {}
        if opt.precompute_use_disk and os.path.exists(scene_feature_dir):
            disk_partial = self._load_frame_cache_from_disk_partial(scene_feature_dir, frame_id, source, opt)
            for key, value in disk_partial.items():
                if isinstance(value, dict):
                    merged = dict(cached.get(key, {}))
                    merged.update(value)
                    cached[key] = merged

        if not self._component_complete(cached, "semantic"):
            self._ensure_semantic_runner(opt)
            sem_outputs = self._semantic_runner(opt).process_frame(
                scene_path=scene_path,
                scene_feature_dir=scene_feature_dir,
                frame_id=frame_id,
                save_vis=save_vis,
                write_disk=write_disk,
            )
            cached["semantic"] = {cam: sem_outputs[cam]["semantic"] for cam in sem_outputs}
            cached["instance"] = {cam: sem_outputs[cam]["instance"] for cam in sem_outputs}
        else:
            self._fill_missing_instances(cached)

        if source == "depth" and not self._component_complete(cached, "depth"):
            self._ensure_depth_runner(opt)
            depth_runner = self._depth_runner(opt)
            prev_scale = self._vggt_state.get(os.path.basename(os.path.normpath(scene_path)), 20.0)
            if str(opt.depth_prefix).lower() == "vggt":
                vggt_outputs = depth_runner.process_frame(
                    scene_path=scene_path,
                    scene_feature_dir=scene_feature_dir,
                    frame_id=frame_id,
                    prev_scale=prev_scale,
                    write_disk=write_disk,
                    save_vis=save_vis,
                    depth_prefix=opt.depth_prefix,
                )
                self._vggt_state[os.path.basename(os.path.normpath(scene_path))] = vggt_outputs["prev_scale"]
                cached["depth"] = vggt_outputs["depths"]
            else:
                ma_outputs = depth_runner.process_frame(
                    scene_path=scene_path,
                    scene_feature_dir=scene_feature_dir,
                    frame_id=frame_id,
                    write_disk=write_disk,
                    save_vis=save_vis,
                )
                cached["depth"] = ma_outputs["depths"]
        self._put_frame_cache(os.path.basename(os.path.normpath(scene_path)), frame_id, cached)
        return cached

    def _ensure_raft_state_from_prev_frame(
        self,
        scene_path,
        scene_feature_dir,
        frame_id,
        source,
        opt,
        write_disk,
        save_vis,
    ):
        if source != "depth" or int(frame_id) <= 0:
            return True
        scene_name = os.path.basename(os.path.normpath(scene_path))
        if (self._raft_state.get(scene_name) is not None
                and self._raft_state_frame.get(scene_name) == int(frame_id) - 1):
            return True
        self._raft_state.pop(scene_name, None)
        prev_id = int(frame_id) - 1
        prev_cache = self._load_or_compute_frame_cache_without_raft(
            scene_path=scene_path,
            scene_feature_dir=scene_feature_dir,
            frame_id=prev_id,
            source=source,
            opt=opt,
            write_disk=write_disk,
            save_vis=save_vis,
        )
        if not self._component_complete(prev_cache, "depth"):
            return False
        prev_prev_instances = None
        if prev_id > 0:
            prev_prev_cache = self._load_or_compute_frame_cache_without_raft(
                scene_path=scene_path,
                scene_feature_dir=scene_feature_dir,
                frame_id=prev_id - 1,
                source=source,
                opt=opt,
                write_disk=write_disk,
                save_vis=save_vis,
            )
            prev_prev_instances = prev_prev_cache.get("instance")
        self._ensure_raft_runner(opt)
        raft_outputs = self._raft_runner.process_frame(
            scene_path=scene_path,
            scene_feature_dir=scene_feature_dir,
            frame_id=prev_id,
            state=None,
            depth_overrides=prev_cache.get("depth"),
            prev_instance_overrides=prev_prev_instances,
            **self._raft_runner_kwargs(opt),
        )
        self._raft_state[scene_name] = raft_outputs["state"]
        self._raft_state_frame[scene_name] = prev_id
        return self._raft_state.get(scene_name) is not None

    def ensure_scene_features(self, scene_path, source, opt):
        scene_name = os.path.basename(os.path.normpath(scene_path))
        scene_feature_dir = precomputed_scene_dir(scene_path, opt.feature_root)
        lock = self._get_scene_lock(scene_name)
        with lock:
            _validate_openseed_cache(scene_feature_dir, opt)
            if (not opt.enable_testtime_precompute) and (
                not self.is_scene_ready(scene_feature_dir, source, opt, scene_path=scene_path)
            ):
                raise FileNotFoundError(
                    f"Missing precomputed features for {scene_name}: {scene_feature_dir}. "
                    "Enable --enable_testtime_precompute or pre-generate features."
                )
            if self.is_scene_ready(scene_feature_dir, source, opt, scene_path=scene_path):
                return scene_feature_dir

            write_disk = bool(opt.precompute_write_disk)
            save_vis = bool(opt.precompute_save_vis)
            if _should_materialize_feature_dirs(opt):
                os.makedirs(scene_feature_dir, exist_ok=True)
                if write_disk:
                    self._ensure_scene_feature_dirs(scene_feature_dir, opt, source)
                elif save_vis:
                    _ensure_semantic_output_dirs(scene_feature_dir, opt)
                    if source == "depth":
                        _ensure_depth_vis_dirs(scene_feature_dir, opt)
                        _ensure_raft_vis_dirs(scene_feature_dir, opt)
            lock_file = _scene_lock_file(scene_name, scene_feature_dir, opt)
            _touch(lock_file)
            try:
                self._ensure_all_runners(opt)
                semantic_runner = self._semantic_runner(opt)
                max_frame = int((len(os.listdir(os.path.join(scene_path, "0"))) - 1) / 2)
                prev_sem_outputs = None
                prev_scale = self._vggt_state.get(scene_name, 20.0)
                for frame_id in range(0, max_frame):
                    sem_outputs = semantic_runner.process_frame(
                        scene_path=scene_path,
                        scene_feature_dir=scene_feature_dir,
                        frame_id=frame_id,
                        save_vis=save_vis,
                        write_disk=write_disk,
                    )
                    if source == "depth":
                        depth_runner = self._depth_runner(opt)
                        if str(opt.depth_prefix).lower() == "vggt":
                            depth_outputs = depth_runner.process_frame(
                                scene_path=scene_path,
                                scene_feature_dir=scene_feature_dir,
                                frame_id=frame_id,
                                prev_scale=prev_scale,
                                write_disk=write_disk,
                                save_vis=save_vis,
                                depth_prefix=opt.depth_prefix,
                            )
                            prev_scale = depth_outputs["prev_scale"]
                            depth_overrides = depth_outputs["depths"]
                        else:
                            depth_outputs = depth_runner.process_frame(
                                scene_path=scene_path,
                                scene_feature_dir=scene_feature_dir,
                                frame_id=frame_id,
                                write_disk=write_disk,
                                save_vis=save_vis,
                            )
                            depth_overrides = depth_outputs["depths"]
                        prev_instances = None
                        if prev_sem_outputs is not None:
                            prev_instances = {
                                cam: prev_sem_outputs[cam]["instance"] for cam in prev_sem_outputs
                            }
                        raft_outputs = self._raft_runner.process_frame(
                            scene_path=scene_path,
                            scene_feature_dir=scene_feature_dir,
                            frame_id=frame_id,
                            state=self._raft_state.get(scene_name),
                            depth_overrides=depth_overrides,
                            prev_instance_overrides=prev_instances,
                            **self._raft_runner_kwargs(opt),
                        )
                        self._raft_state[scene_name] = raft_outputs["state"]
                        self._raft_state_frame[scene_name] = int(frame_id)
                    prev_sem_outputs = sem_outputs
                self._vggt_state[scene_name] = prev_scale
            finally:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
        return scene_feature_dir

    def _ensure_scene_feature_dirs(self, scene_feature_dir, opt, source):
        for cam in range(6):
            os.makedirs(os.path.join(scene_feature_dir, f"{opt.semantic_prefix}_{cam}"), exist_ok=True)
            if source == "depth":
                os.makedirs(os.path.join(scene_feature_dir, f"{opt.depth_prefix}_{cam}"), exist_ok=True)

    def ensure_frame_features(self, scene_path, frame_id, source, opt):
        scene_name = os.path.basename(os.path.normpath(scene_path))
        scene_feature_dir = precomputed_scene_dir(scene_path, opt.feature_root)
        lock = self._get_scene_lock(scene_name)
        with lock:
            _validate_openseed_cache(scene_feature_dir, opt)
            frame_id = int(frame_id)
            if source == "depth" and opt.depth_prefix == "mapanything":
                import json
                recipe_file = Path(scene_feature_dir, "mapanything_recipe.json")
                if recipe_file.exists():
                    actual = json.loads(recipe_file.read_text())
                    expected = {"model": opt.mapanything_model_path, "revision": opt.mapanything_revision,
                                "scope": opt.mapanything_scope, "apply_mask": bool(opt.mapanything_apply_mask),
                                "mask_edges": bool(opt.mapanything_mask_edges), "matmul_tf32": True,
                                "cudnn_benchmark": False, "encoder_batch": 6, "head_batch": 6}
                    if any(actual.get(k) != v for k, v in expected.items()):
                        raise ValueError(f"Incompatible MapAnything cache: {recipe_file}; use a fresh feature_root")
                elif any(Path(scene_feature_dir, f"mapanything_{c}", f"{frame_id:02d}.npy").exists() for c in range(6)):
                    raise ValueError(f"Depth cache has no provenance: {scene_feature_dir}; use a fresh feature_root")
            profile = {
                "status": "unknown",
                "total_s": 0.0,
                "semantic_s": 0.0,
                "depth_s": 0.0,
                "raft_s": 0.0,
                "raft_warmup_s": 0.0,
            }
            frame_cache = self.get_cached_frame_features(scene_path, frame_id)
            sem_depth_from_cache = False
            if frame_cache is not None and source != "depth":
                profile["status"] = "memory_hit"
                self._frame_profiles.setdefault((scene_name, frame_id), profile)
                return scene_feature_dir
            if (
                frame_cache is not None
                and source == "depth"
                and self._component_complete(frame_cache, "semantic")
                and self._component_complete(frame_cache, "depth")
            ):
                if frame_id == 0 or self._component_complete(frame_cache, "dynamic"):
                    profile["status"] = "memory_hit"
                    self._frame_profiles.setdefault((scene_name, frame_id), profile)
                    return scene_feature_dir
                sem_depth_from_cache = True
                profile["status"] = "memory_hit"
            readiness = "not_ready"
            if opt.precompute_use_disk:
                readiness = self._frame_readiness_state(scene_feature_dir, frame_id, source, opt)
            if opt.precompute_use_disk and readiness == "full_ready" and source == "lidar":
                profile["status"] = "disk_hit_full"
                frame_cache = self._load_frame_cache_from_disk(
                    scene_feature_dir=scene_feature_dir,
                    frame_id=frame_id,
                    source=source,
                    opt=opt,
                )
                self._put_frame_cache(scene_name, frame_id, frame_cache)
                self._frame_profiles[(scene_name, frame_id)] = profile
                return scene_feature_dir
            if opt.precompute_use_disk and readiness == "full_ready" and source == "depth" and not sem_depth_from_cache:
                frame_cache = self._load_frame_cache_from_disk(
                    scene_feature_dir=scene_feature_dir,
                    frame_id=frame_id,
                    source=source,
                    opt=opt,
                )
                sem_depth_from_cache = True
                profile["status"] = "disk_hit_depth"
                if getattr(opt, "precompute_cache_only", False):
                    self._put_frame_cache(scene_name, frame_id, frame_cache)
                    self._frame_profiles[(scene_name, frame_id)] = profile
                    return scene_feature_dir
            if getattr(opt, "precompute_cache_only", False):
                raise FileNotFoundError(
                    f"Cache-only inference requires all frame features: {scene_feature_dir}, frame {frame_id:02d}"
                )
            if not opt.enable_testtime_precompute:
                raise FileNotFoundError(
                    f"Missing precomputed frame features for {scene_name} frame {frame_id:02d}: {scene_feature_dir}. "
                    "Enable --enable_testtime_precompute or pre-generate features."
                )
            write_disk = bool(opt.precompute_write_disk)
            save_vis = bool(opt.precompute_save_vis)
            if _should_materialize_feature_dirs(opt):
                os.makedirs(scene_feature_dir, exist_ok=True)
                if write_disk:
                    self._ensure_scene_feature_dirs(scene_feature_dir, opt, source)
                elif save_vis:
                    _ensure_semantic_output_dirs(scene_feature_dir, opt)
                    if source == "depth":
                        _ensure_depth_vis_dirs(scene_feature_dir, opt)
                        _ensure_raft_vis_dirs(scene_feature_dir, opt)
            lock_file = _scene_lock_file(scene_name, scene_feature_dir, opt)
            _touch(lock_file)
            if not sem_depth_from_cache:
                profile["status"] = "computed"
            if opt.precompute_use_disk and readiness == "partial_ready":
                profile["status"] = "disk_hit_partial_recovered"
            total_start = time.time()
            try:
                openseed_start = time.time()
                if not sem_depth_from_cache:
                    frame_cache = {}
                if opt.precompute_use_disk and readiness == "partial_ready":
                    frame_cache = self._load_frame_cache_from_disk_partial(
                        scene_feature_dir=scene_feature_dir,
                        frame_id=frame_id,
                        source=source,
                        opt=opt,
                    )
                if self._component_complete(frame_cache, "semantic"):
                    self._fill_missing_instances(frame_cache)
                    sem_outputs = {
                        cam: {
                            "semantic": frame_cache["semantic"][cam],
                            "instance": frame_cache["instance"][cam],
                        }
                        for cam in range(6)
                    }
                else:
                    self._ensure_semantic_runner(opt)
                    sem_outputs = self._semantic_runner(opt).process_frame(
                        scene_path=scene_path,
                        scene_feature_dir=scene_feature_dir,
                        frame_id=frame_id,
                        save_vis=save_vis,
                        write_disk=write_disk,
                    )
                    frame_cache["semantic"] = {cam: sem_outputs[cam]["semantic"] for cam in sem_outputs}
                    frame_cache["instance"] = {cam: sem_outputs[cam]["instance"] for cam in sem_outputs}
                profile["semantic_s"] = time.time() - openseed_start
                if source == "depth":
                    prev_scale = self._vggt_state.get(scene_name, 20.0)
                    vggt_start = time.time()
                    if self._component_complete(frame_cache, "depth"):
                        depths = frame_cache["depth"]
                        if save_vis:
                            _save_depth_frame_vis(depths, scene_feature_dir, frame_id, opt.depth_prefix)
                    else:
                        self._ensure_depth_runner(opt)
                        depth_runner = self._depth_runner(opt)
                        if str(opt.depth_prefix).lower() == "vggt":
                            vggt_outputs = depth_runner.process_frame(
                                scene_path=scene_path,
                                scene_feature_dir=scene_feature_dir,
                                frame_id=frame_id,
                                prev_scale=prev_scale,
                                write_disk=write_disk,
                                save_vis=save_vis,
                                depth_prefix=opt.depth_prefix,
                            )
                            self._vggt_state[scene_name] = vggt_outputs["prev_scale"]
                            depths = vggt_outputs["depths"]
                        else:
                            ma_outputs = depth_runner.process_frame(
                                scene_path=scene_path,
                                scene_feature_dir=scene_feature_dir,
                                frame_id=frame_id,
                                write_disk=write_disk,
                                save_vis=save_vis,
                            )
                            depths = ma_outputs["depths"]
                        frame_cache["depth"] = depths
                    profile["depth_s"] = time.time() - vggt_start
                    prev_instances = None
                    if frame_id > 0:
                        prev_cached = self.get_cached_frame_features(scene_path, frame_id - 1)
                        if prev_cached is None or not self._component_complete(prev_cached, "depth"):
                            prev_cached = self._load_or_compute_frame_cache_without_raft(
                                scene_path=scene_path,
                                scene_feature_dir=scene_feature_dir,
                                frame_id=frame_id - 1,
                                source=source,
                                opt=opt,
                                write_disk=write_disk,
                                save_vis=save_vis,
                            )
                        self._fill_missing_instances(prev_cached)
                        prev_instances = prev_cached.get("instance")
                        if not self._component_complete(prev_cached, "depth"):
                            raise RuntimeError(
                                f"RAFT strict mode: cached t-1 data is incomplete for {scene_name} frame {frame_id - 1:02d}. "
                                "Expected depth in memory cache."
                            )
                        if self._raft_state.get(scene_name) is None:
                            ok = self._ensure_raft_state_from_prev_frame(
                                scene_path=scene_path,
                                scene_feature_dir=scene_feature_dir,
                                frame_id=frame_id,
                                source=source,
                                opt=opt,
                                write_disk=write_disk,
                                save_vis=save_vis,
                            )
                            if not ok:
                                raise RuntimeError(
                                    f"RAFT strict mode recovery failed for {scene_name} before frame {frame_id:02d}. "
                                    "Unable to bootstrap temporal state from disk/test-time replay."
                                )
                    self._ensure_raft_runner(opt)
                    raft_start = time.time()
                    raft_outputs = self._raft_runner.process_frame(
                        scene_path=scene_path,
                        scene_feature_dir=scene_feature_dir,
                        frame_id=frame_id,
                        state=self._raft_state.get(scene_name),
                        depth_overrides=depths,
                        prev_instance_overrides=prev_instances,
                        **self._raft_runner_kwargs(opt),
                    )
                    profile["raft_s"] = time.time() - raft_start
                    self._raft_state[scene_name] = raft_outputs["state"]
                    self._raft_state_frame[scene_name] = int(frame_id)
                    frame_cache["dynamic"] = raft_outputs.get("dynamic_masks", {})
                self._put_frame_cache(scene_name, frame_id, frame_cache)
            finally:
                profile["total_s"] = time.time() - total_start
                self._frame_profiles[(scene_name, frame_id)] = profile
                if os.path.exists(lock_file):
                    os.remove(lock_file)
        return scene_feature_dir

    def _load_frame_cache_from_disk(self, scene_feature_dir, frame_id, source, opt):
        frame_name = f"{int(frame_id):0>2d}"
        frame_cache = {
            "semantic": {},
            "instance": {},
        }
        for cam in range(6):
            sem_path = os.path.join(scene_feature_dir, f"{opt.semantic_prefix}_{cam}", f"{frame_name}.png")
            frame_cache["semantic"][cam] = self._read_png(sem_path)
            frame_cache["instance"][cam] = self._read_instance_cam(
                scene_feature_dir, opt, cam, frame_name, frame_cache["semantic"][cam]
            )
        if source == "depth":
            frame_cache["depth"] = {}
            for cam in range(6):
                depth_path = os.path.join(scene_feature_dir, f"{opt.depth_prefix}_{cam}", f"{frame_name}.{opt.depth_ext}")
                frame_cache["depth"][cam] = self._read_depth(depth_path, opt.depth_ext)
            if getattr(opt, "precompute_cache_only", False) and int(frame_id) > 0:
                frame_cache["dynamic"] = {}
                for cam in range(6):
                    mask_path = os.path.join(scene_feature_dir, f"{opt.dynamic_mask_prefix}_{cam}",
                                             f"{frame_name}{opt.dynamic_mask_suffix}")
                    frame_cache["dynamic"][cam] = self._read_png(mask_path)
        return frame_cache

    @staticmethod
    def _read_png(path):
        import cv2
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(f"Failed to read cached png: {path}")
        return arr

    @staticmethod
    def _read_depth(path, depth_ext):
        if depth_ext == "npy":
            return np.load(path)
        raise ValueError(f"Unsupported depth extension for cache load: {depth_ext}")

    def prefetch_scene(self, scene_path, source, opt):
        if not opt.precompute_prefetch:
            return None
        scene_name = os.path.basename(os.path.normpath(scene_path))
        if scene_name in self._prefetch_futures:
            f = self._prefetch_futures[scene_name]
            if not f.done():
                return f

        def _task():
            scene_feature_dir = precomputed_scene_dir(scene_path, opt.feature_root)
            self.is_scene_ready(scene_feature_dir, source, opt, scene_path=scene_path)
            return scene_feature_dir

        future = self._prefetch_executor.submit(_task)
        self._prefetch_futures[scene_name] = future
        return future

    def prefetch_frame(self, scene_path, frame_id, source, opt):
        if not opt.precompute_prefetch:
            return None
        scene_name = os.path.basename(os.path.normpath(scene_path))
        frame_key = (scene_name, int(frame_id))
        if frame_key in self._frame_prefetch_futures:
            f = self._frame_prefetch_futures[frame_key]
            if not f.done():
                return f

        def _task():
            scene_feature_dir = precomputed_scene_dir(scene_path, opt.feature_root)
            self.ensure_frame_features(scene_path, frame_id, source, opt)
            return scene_feature_dir

        future = self._prefetch_executor.submit(_task)
        self._frame_prefetch_futures[frame_key] = future
        return future

    def wait_prefetch(self, scene_path):
        scene_name = os.path.basename(os.path.normpath(scene_path))
        f = self._prefetch_futures.get(scene_name)
        if f is not None and isinstance(f, Future):
            return f.result()
        return None

    def wait_prefetch_frame(self, scene_path, frame_id):
        scene_name = os.path.basename(os.path.normpath(scene_path))
        frame_key = (scene_name, int(frame_id))
        f = self._frame_prefetch_futures.get(frame_key)
        if f is not None and isinstance(f, Future):
            return f.result()
        return None

    def close(self):
        try:
            self._prefetch_executor.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            self._prefetch_executor.shutdown(wait=True)
        self._prefetch_futures.clear()
        self._frame_prefetch_futures.clear()
        self._vggt_state.clear()
        self._raft_state.clear()
        self._raft_state_frame.clear()
