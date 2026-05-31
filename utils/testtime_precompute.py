import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

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


class ScenePrecomputeManager:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._scene_locks = {}
        self._scene_locks_guard = threading.Lock()
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1)
        self._prefetch_futures = {}
        self._frame_prefetch_futures = {}
        self._openseed_runner = None
        self._vggt_runner = None
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
                if self._openseed_runner is None:
                    self._openseed_runner = OpenSeeDRunner(use_inference_mode=opt.precompute_inference_mode)
                if self._vggt_runner is None:
                    self._vggt_runner = VGGTRunner(use_inference_mode=opt.precompute_inference_mode)
                if self._raft_runner is None:
                    self._raft_runner = RAFTRunner(use_inference_mode=opt.precompute_inference_mode)
            finally:
                os.chdir(cwd)

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
                self._openseed_runner.process_scene(
                    scene_path=scene_path,
                    scene_feature_dir=scene_feature_dir,
                    save_vis=opt.precompute_save_vis,
                )
                self._vggt_runner.process_scene(
                    scene_path=scene_path,
                    scene_feature_dir=scene_feature_dir,
                    frame_step=opt.precompute_frame_step,
                )
                self._raft_runner.process_scene(
                    scene_path=scene_path,
                    scene_feature_dir=scene_feature_dir,
                    frame_step=opt.precompute_frame_step,
                )
                _touch(os.path.join(scene_feature_dir, ".ready"))
            finally:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
        return scene_feature_dir

    def _ensure_scene_feature_dirs(self, scene_feature_dir, opt, source):
        for cam in range(6):
            os.makedirs(os.path.join(scene_feature_dir, f"{opt.semantic_prefix}_{cam}"), exist_ok=True)
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
                openseed_start = time.time()
                sem_outputs = self._openseed_runner.process_frame(
                    scene_path=scene_path,
                    scene_feature_dir=scene_feature_dir,
                    frame_id=frame_id,
                    save_vis=opt.precompute_save_vis,
                    write_disk=bool(opt.precompute_write_disk),
                )
                profile["semantic_s"] = time.time() - openseed_start
                frame_cache = {
                    "semantic": {cam: sem_outputs[cam]["semantic"] for cam in sem_outputs},
                    "instance": {cam: sem_outputs[cam]["instance"] for cam in sem_outputs},
                }
                if source == "depth":
                    prev_scale = self._vggt_state.get(scene_name, 20.0)
                    vggt_start = time.time()
                    vggt_outputs = self._vggt_runner.process_frame(
                        scene_path=scene_path,
                        scene_feature_dir=scene_feature_dir,
                        frame_id=frame_id,
                        prev_scale=prev_scale,
                        write_disk=bool(opt.precompute_write_disk),
                    )
                    profile["depth_s"] = time.time() - vggt_start
                    self._vggt_state[scene_name] = vggt_outputs["prev_scale"]
                    frame_cache["depth"] = vggt_outputs["depths"]
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
                        depth_overrides=vggt_outputs["depths"],
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
            if opt.precompute_use_disk:
                self.is_frame_ready(scene_feature_dir, frame_id, source, opt)
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
