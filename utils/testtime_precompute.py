import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _repo_root():
    return Path(__file__).resolve().parents[1]

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
        reqs += [f"{dynamic_mask_prefix}_{cam}" for cam in range(6)]
    return reqs


def frame_feature_requirements(
    frame_id,
    source="lidar",
    semantic_prefix="openseed",
    depth_prefix="vggt",
    depth_ext="npy",
    dynamic_mask_prefix="raft",
    dynamic_mask_suffix="_5mask.png",
):
    frame_name = f"{int(frame_id):0>2d}"
    reqs = []
    for cam in range(6):
        reqs.append(f"{semantic_prefix}_{cam}/{frame_name}.png")
        reqs.append(f"{semantic_prefix}_{cam}/{frame_name}_instance.png")
    if source == "depth":
        for cam in range(6):
            reqs.append(f"{depth_prefix}_{cam}/{frame_name}.{depth_ext}")
        if int(frame_id) > 0:
            for cam in range(6):
                reqs.append(f"{dynamic_mask_prefix}_{cam}/{frame_name}{dynamic_mask_suffix}")
    return reqs


def _lazy_import_openseed():
    import importlib.util
    submodule_root = _repo_root() / "submodules" / "OpenSeeD"
    if str(submodule_root) not in sys.path:
        sys.path.insert(0, str(submodule_root))
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


class OpenSeeDRunner:
    def __init__(self, use_inference_mode=True):
        mod = _lazy_import_openseed()
        self.runtime = mod.build_openseed_runtime(
            conf_file="submodules/OpenSeeD/configs/openseed/openseed_swint_lang.yaml",
            weight_path="model_state_dict_swint_51.2ap.pt",
            use_inference_mode=use_inference_mode,
        )
        self._mod = mod

    def process_scene(self, scene_path, scene_feature_dir, save_vis=False):
        return self._mod.process_scene(
            self.runtime,
            scene_in=scene_path,
            scene_out=scene_feature_dir,
            save_vis=save_vis,
        )

    def process_frame(self, scene_path, scene_feature_dir, frame_id, save_vis=False, write_disk=True):
        return self._mod.process_frame_openseed(
            self.runtime,
            scene_in=scene_path,
            scene_out=scene_feature_dir,
            frame=int(frame_id),
            save_vis=save_vis,
            write_disk=write_disk,
        )


class VGGTRunner:
    def __init__(self, use_inference_mode=True):
        mod = _lazy_import_vggt()
        self.runtime = mod.build_vggt_runtime(device="cuda")
        self.use_inference_mode = use_inference_mode
        self._mod = mod

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

    def process_frame(self, scene_path, scene_feature_dir, frame_id, prev_scale, write_disk=True):
        return self._mod.process_frame_vggt_api(
            self.runtime,
            scene_in=scene_path,
            scene_out=scene_feature_dir,
            frame=int(frame_id),
            prev_scale=prev_scale,
            use_inference_mode=self.use_inference_mode,
            write_disk=write_disk,
            write_png=False,
        )


class RAFTRunner:
    def __init__(self, use_inference_mode=True):
        mod = _lazy_import_raft()
        self.runtime = mod.build_raft_runtime(model_path="submodules/RAFT/raft-things.pth")
        self.use_inference_mode = use_inference_mode
        self._mod = mod

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

    def process_frame(
        self,
        scene_path,
        scene_feature_dir,
        frame_id,
        state,
        write_disk=True,
        depth_overrides=None,
        prev_instance_overrides=None,
    ):
        return self._mod.process_frame_raft_api(
            self.runtime,
            scene_in=scene_path,
            scene_out=scene_feature_dir,
            frame=int(frame_id),
            state=state,
            use_inference_mode=self.use_inference_mode,
            write_disk=write_disk,
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

    def __init__(
        self,
        semantic_prefix="rexomni",
        model_path="IDEA-Research/Rex-Omni",
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
        self.sam_box_batch = _env_flag("TTOCC_SAM_BOX_BATCH", default=False)
        checkpoint_path = sam_checkpoint
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = str(_repo_root() / checkpoint_path)
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
                if category in self._mod.INDEX_MAPPING:
                    semantic[mask] = self._mod.INDEX_MAPPING[category]
                instance_id = min(next_instance_id, 255)
                instance[mask] = instance_id
                next_instance_id += 1
                mask_idx += 1
        return semantic, instance

    @staticmethod
    def _load_rgb(image_path):
        image = Image.open(image_path).convert("RGB")
        return image, np.array(image)

    def process_frame(self, scene_path, scene_feature_dir, frame_id, write_disk=True):
        frame_name = f"{int(frame_id):0>2d}"
        image_paths = [os.path.join(scene_path, f"{cam}", f"{frame_name}.jpg") for cam in range(6)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            loaded = list(pool.map(self._load_rgb, image_paths))
        images = [item[0] for item in loaded]
        img_arrays = [item[1] for item in loaded]
        inference_ctx = torch.inference_mode if self.use_inference_mode else torch.no_grad
        with inference_ctx():
            batched_results = self._mod.detect_objects_with_rex_batch(
                self.rex_model,
                images,
                self.categories,
            )

        frame_outputs = {}
        for cam in range(6):
            predictions = batched_results[cam]["extracted_predictions"]
            semantic, instance = self._build_semantic_instance(img_arrays[cam], predictions)
            frame_outputs[cam] = {"semantic": semantic, "instance": instance}
            if write_disk:
                out_dir = os.path.join(scene_feature_dir, f"{self.semantic_prefix}_{cam}")
                os.makedirs(out_dir, exist_ok=True)
                Image.fromarray(semantic).save(os.path.join(out_dir, f"{frame_name}.png"))
                Image.fromarray(instance).save(os.path.join(out_dir, f"{frame_name}_instance.png"))
        return frame_outputs


class MapAnythingRunner:
    def __init__(self, depth_prefix="mapanything", model_path="facebook/map-anything", use_inference_mode=True):
        mod = _lazy_import_mapanything()
        self._mod = mod
        self.depth_prefix = depth_prefix
        self.use_inference_mode = use_inference_mode
        self.model = mod.MapAnything.from_pretrained(model_path).to("cuda")
        self.model.eval()
        self.device = "cuda"
        self._scene_static = {}
        self._pose_cache = {}
        self.memory_efficient_inference = _env_flag("TTOCC_MAPANY_MEMORY_EFFICIENT", default=False)

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

    def process_frame(self, scene_path, scene_feature_dir, frame_id, write_disk=True):
        frame_id = int(frame_id)
        frame_name = f"{frame_id:0>2d}"
        meta = self._scene_meta(scene_path)
        image_names = [os.path.join(scene_path, f"{cam}", f"{frame_name}.jpg") for cam in range(6)]
        images = self._mod.load_and_preprocess_images(image_names).to(self.device)
        cam2world_list = self._load_frame_poses(scene_path, frame_name, meta["old_world2new_world"])
        amp_dtype = "bf16" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "fp16"
        inference_ctx = torch.inference_mode if self.use_inference_mode else torch.no_grad
        with inference_ctx():
            depth_map = self._mod.run_mapanything_depth_only(
                self.model,
                images,
                cam2world_list,
                meta["intrinsics"],
                self.model.encoder.data_norm_type,
                memory_efficient_inference=self.memory_efficient_inference,
                use_amp=True,
                amp_dtype=amp_dtype,
            )

        depths = {}
        for cam in range(6):
            depth = depth_map[cam].astype(np.float32)
            depths[cam] = depth
            if write_disk:
                out_dir = os.path.join(scene_feature_dir, f"{self.depth_prefix}_{cam}")
                os.makedirs(out_dir, exist_ok=True)
                np.save(os.path.join(out_dir, frame_name), depth)
        return {"depths": depths}


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
        self._cache_keep_frames = 1
        self._vggt_state = {}
        self._raft_state = {}
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

    def _ready_flag_path(self, scene_feature_dir, frame_id):
        frame_name = f"{int(frame_id):0>2d}"
        return os.path.join(scene_feature_dir, ".ready_frames", f"{frame_name}.ok")

    def _mark_frame_ready(self, scene_feature_dir, frame_id):
        _touch(self._ready_flag_path(scene_feature_dir, frame_id))

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

    def _ensure_runners(self, opt):
        with self._runner_lock:
            cwd = os.getcwd()
            try:
                os.chdir(str(_repo_root()))
                semantic_key = str(opt.semantic_prefix).lower()
                if semantic_key == "openseed":
                    if self._openseed_runner is None:
                        self._openseed_runner = OpenSeeDRunner(
                            use_inference_mode=opt.precompute_inference_mode
                        )
                elif semantic_key == "rexomni":
                    if self._rexomni_runner is None:
                        self._rexomni_runner = RexOmniRunner(
                            semantic_prefix=opt.semantic_prefix,
                            use_inference_mode=opt.precompute_inference_mode,
                        )
                else:
                    raise ValueError(f"Unsupported semantic_prefix: {opt.semantic_prefix}")

                if opt.variant == "depth":
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
                            )
                    else:
                        raise ValueError(f"Unsupported depth_prefix: {opt.depth_prefix}")
                    if self._raft_runner is None:
                        self._raft_runner = RAFTRunner(use_inference_mode=opt.precompute_inference_mode)
            finally:
                os.chdir(cwd)

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

    def is_scene_ready(self, scene_feature_dir, source, opt):
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
        ready_flag = os.path.join(scene_feature_dir, ".ready")
        if not os.path.exists(ready_flag):
            return False
        for rel in reqs:
            p = os.path.join(scene_feature_dir, rel)
            if not os.path.exists(p):
                return False
            if os.path.isdir(p):
                if len(os.listdir(p)) == 0:
                    return False
        return True

    def is_frame_ready(self, scene_feature_dir, frame_id, source, opt):
        if not os.path.exists(scene_feature_dir):
            return False
        if not os.path.exists(self._ready_flag_path(scene_feature_dir, frame_id)):
            return False
        reqs = frame_feature_requirements(
            frame_id=frame_id,
            source=source,
            semantic_prefix=opt.semantic_prefix,
            depth_prefix=opt.depth_prefix,
            depth_ext=opt.depth_ext,
            dynamic_mask_prefix=opt.dynamic_mask_prefix,
            dynamic_mask_suffix=opt.dynamic_mask_suffix,
        )
        for rel in reqs:
            if not os.path.exists(os.path.join(scene_feature_dir, rel)):
                return False
        return True

    def ensure_scene_features(self, scene_path, source, opt):
        scene_name = os.path.basename(os.path.normpath(scene_path))
        scene_feature_dir = precomputed_scene_dir(scene_path, opt.feature_root)
        lock = self._get_scene_lock(scene_name)
        with lock:
            if (not opt.enable_testtime_precompute) and (not self.is_scene_ready(scene_feature_dir, source, opt)):
                raise FileNotFoundError(
                    f"Missing precomputed features for {scene_name}: {scene_feature_dir}. "
                    "Enable --enable_testtime_precompute or pre-generate features."
                )
            if self.is_scene_ready(scene_feature_dir, source, opt):
                return scene_feature_dir

            os.makedirs(scene_feature_dir, exist_ok=True)
            lock_file = os.path.join(scene_feature_dir, ".lock")
            _touch(lock_file)
            try:
                self._ensure_runners(opt)
                semantic_runner = self._semantic_runner(opt)
                max_frame = int((len(os.listdir(os.path.join(scene_path, "0"))) - 1) / 2)
                prev_sem_outputs = None
                prev_scale = self._vggt_state.get(scene_name, 20.0)
                for frame_id in range(0, max_frame):
                    sem_outputs = semantic_runner.process_frame(
                        scene_path=scene_path,
                        scene_feature_dir=scene_feature_dir,
                        frame_id=frame_id,
                        write_disk=bool(opt.precompute_write_disk),
                    )
                    if source == "depth":
                        depth_runner = self._depth_runner(opt)
                        if str(opt.depth_prefix).lower() == "vggt":
                            depth_outputs = depth_runner.process_frame(
                                scene_path=scene_path,
                                scene_feature_dir=scene_feature_dir,
                                frame_id=frame_id,
                                prev_scale=prev_scale,
                                write_disk=bool(opt.precompute_write_disk),
                            )
                            prev_scale = depth_outputs["prev_scale"]
                            depth_overrides = depth_outputs["depths"]
                        else:
                            depth_outputs = depth_runner.process_frame(
                                scene_path=scene_path,
                                scene_feature_dir=scene_feature_dir,
                                frame_id=frame_id,
                                write_disk=bool(opt.precompute_write_disk),
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
                            write_disk=bool(opt.precompute_write_disk),
                            depth_overrides=depth_overrides,
                            prev_instance_overrides=prev_instances,
                        )
                        self._raft_state[scene_name] = raft_outputs["state"]
                    prev_sem_outputs = sem_outputs
                self._vggt_state[scene_name] = prev_scale
                if opt.precompute_write_disk:
                    _touch(os.path.join(scene_feature_dir, ".ready"))
            finally:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
        return scene_feature_dir

    def _ensure_scene_feature_dirs(self, scene_feature_dir, opt, source):
        for cam in range(6):
            os.makedirs(os.path.join(scene_feature_dir, f"{opt.semantic_prefix}_{cam}"), exist_ok=True)
            if source == "depth":
                os.makedirs(os.path.join(scene_feature_dir, f"{opt.depth_prefix}_{cam}"), exist_ok=True)
                os.makedirs(os.path.join(scene_feature_dir, f"{opt.dynamic_mask_prefix}_{cam}"), exist_ok=True)
        os.makedirs(os.path.join(scene_feature_dir, ".ready_frames"), exist_ok=True)

    def ensure_frame_features(self, scene_path, frame_id, source, opt):
        scene_name = os.path.basename(os.path.normpath(scene_path))
        scene_feature_dir = precomputed_scene_dir(scene_path, opt.feature_root)
        lock = self._get_scene_lock(scene_name)
        with lock:
            frame_id = int(frame_id)
            profile = {
                "status": "unknown",
                "total_s": 0.0,
                "semantic_s": 0.0,
                "depth_s": 0.0,
                "raft_s": 0.0,
                "raft_warmup_s": 0.0,
            }
            frame_cache = self.get_cached_frame_features(scene_path, frame_id)
            if frame_cache is not None:
                profile["status"] = "memory_hit"
                self._frame_profiles[(scene_name, frame_id)] = profile
                return scene_feature_dir
            if opt.precompute_use_disk and self.is_frame_ready(scene_feature_dir, frame_id, source, opt):
                profile["status"] = "disk_hit"
                frame_cache = self._load_frame_cache_from_disk(
                    scene_feature_dir=scene_feature_dir,
                    frame_id=frame_id,
                    source=source,
                    opt=opt,
                )
                self._put_frame_cache(scene_name, frame_id, frame_cache)
                self._frame_profiles[(scene_name, frame_id)] = profile
                return scene_feature_dir
            if not opt.enable_testtime_precompute:
                raise FileNotFoundError(
                    f"Missing precomputed frame features for {scene_name} frame {frame_id:02d}: {scene_feature_dir}. "
                    "Enable --enable_testtime_precompute or pre-generate features."
                )
            os.makedirs(scene_feature_dir, exist_ok=True)
            self._ensure_scene_feature_dirs(scene_feature_dir, opt, source)
            lock_file = os.path.join(scene_feature_dir, ".lock")
            _touch(lock_file)
            profile["status"] = "computed"
            total_start = time.time()
            try:
                self._ensure_runners(opt)
                semantic_runner = self._semantic_runner(opt)
                openseed_start = time.time()
                sem_outputs = semantic_runner.process_frame(
                    scene_path=scene_path,
                    scene_feature_dir=scene_feature_dir,
                    frame_id=frame_id,
                    write_disk=bool(opt.precompute_write_disk),
                )
                profile["semantic_s"] = time.time() - openseed_start
                frame_cache = {
                    "semantic": {cam: sem_outputs[cam]["semantic"] for cam in sem_outputs},
                    "instance": {cam: sem_outputs[cam]["instance"] for cam in sem_outputs},
                }
                if source == "depth":
                    prev_scale = self._vggt_state.get(scene_name, 20.0)
                    depth_runner = self._depth_runner(opt)
                    vggt_start = time.time()
                    if str(opt.depth_prefix).lower() == "vggt":
                        vggt_outputs = depth_runner.process_frame(
                            scene_path=scene_path,
                            scene_feature_dir=scene_feature_dir,
                            frame_id=frame_id,
                            prev_scale=prev_scale,
                            write_disk=bool(opt.precompute_write_disk),
                        )
                        self._vggt_state[scene_name] = vggt_outputs["prev_scale"]
                        depths = vggt_outputs["depths"]
                    else:
                        ma_outputs = depth_runner.process_frame(
                            scene_path=scene_path,
                            scene_feature_dir=scene_feature_dir,
                            frame_id=frame_id,
                            write_disk=bool(opt.precompute_write_disk),
                        )
                        depths = ma_outputs["depths"]
                    profile["depth_s"] = time.time() - vggt_start
                    frame_cache["depth"] = depths
                    prev_instances = None
                    if frame_id > 0:
                        prev_cached = self.get_cached_frame_features(scene_path, frame_id - 1)
                        if prev_cached is None:
                            raise RuntimeError(
                                f"RAFT strict mode: missing cached t-1 frame for {scene_name} frame {frame_id - 1:02d} "
                                f"before processing frame {frame_id:02d}."
                            )
                        prev_instances = prev_cached.get("instance")
                        prev_depth = prev_cached.get("depth")
                        if prev_instances is None or prev_depth is None:
                            raise RuntimeError(
                                f"RAFT strict mode: cached t-1 data is incomplete for {scene_name} frame {frame_id - 1:02d}. "
                                "Expected both instance and depth in memory cache."
                            )
                        if self._raft_state.get(scene_name) is None:
                            raise RuntimeError(
                                f"RAFT strict mode: missing RAFT temporal state for {scene_name} before frame {frame_id:02d}. "
                                "State should be produced by frame 00..t-1 in-order processing."
                            )
                    raft_start = time.time()
                    raft_outputs = self._raft_runner.process_frame(
                        scene_path=scene_path,
                        scene_feature_dir=scene_feature_dir,
                        frame_id=frame_id,
                        state=self._raft_state.get(scene_name),
                        write_disk=bool(opt.precompute_write_disk),
                        depth_overrides=depths,
                        prev_instance_overrides=prev_instances,
                    )
                    profile["raft_s"] = time.time() - raft_start
                    self._raft_state[scene_name] = raft_outputs["state"]
                    frame_cache["dynamic"] = raft_outputs.get("dynamic_masks", {})
                self._put_frame_cache(scene_name, frame_id, frame_cache)
                if opt.precompute_write_disk:
                    self._mark_frame_ready(scene_feature_dir, frame_id)
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
            ins_path = os.path.join(scene_feature_dir, f"{opt.semantic_prefix}_{cam}", f"{frame_name}_instance.png")
            frame_cache["semantic"][cam] = self._read_png(sem_path)
            frame_cache["instance"][cam] = self._read_png(ins_path)
        if source == "depth":
            frame_cache["depth"] = {}
            for cam in range(6):
                depth_path = os.path.join(scene_feature_dir, f"{opt.depth_prefix}_{cam}", f"{frame_name}.{opt.depth_ext}")
                frame_cache["depth"][cam] = self._read_depth(depth_path, opt.depth_ext)
            if int(frame_id) > 0:
                frame_cache["dynamic"] = {}
                for cam in range(6):
                    dyn_path = os.path.join(
                        scene_feature_dir,
                        f"{opt.dynamic_mask_prefix}_{cam}",
                        f"{frame_name}{opt.dynamic_mask_suffix}",
                    )
                    frame_cache["dynamic"][cam] = self._read_png(dyn_path)
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
            self.is_scene_ready(scene_feature_dir, source, opt)
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
            self._prefetch_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._prefetch_executor.shutdown(wait=False)
        self._prefetch_futures.clear()
        self._frame_prefetch_futures.clear()
        self._vggt_state.clear()
        self._raft_state.clear()
