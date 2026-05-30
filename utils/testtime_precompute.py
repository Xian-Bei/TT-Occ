import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

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


class ScenePrecomputeManager:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._scene_locks = {}
        self._scene_locks_guard = threading.Lock()
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1)
        self._prefetch_futures = {}
        self._openseed_runner = None
        self._vggt_runner = None
        self._raft_runner = None
        self._runner_lock = threading.Lock()

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
            if self.is_scene_ready(scene_feature_dir, source, opt) and (not opt.precompute_force):
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
                torch.cuda.empty_cache()
        return scene_feature_dir

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

    def wait_prefetch(self, scene_path):
        scene_name = os.path.basename(os.path.normpath(scene_path))
        f = self._prefetch_futures.get(scene_name)
        if f is not None and isinstance(f, Future):
            return f.result()
        return None

    def close(self):
        try:
            self._prefetch_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._prefetch_executor.shutdown(wait=False)
