"""Explicit backend settings shared by GPU inference and prefetch threads."""
from contextlib import contextmanager
from functools import wraps
import threading

import torch


_gpu_dispatch_lock = threading.RLock()


@contextmanager
def inference_precision(matmul_tf32=True, autocast=None):
    # Backend flags are process-wide. Hold the same lock for every producer and
    # Gaussian stage so a prefetch thread cannot change another stage's math.
    with _gpu_dispatch_lock:
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        previous_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.benchmark = False
        try:
            if autocast is None:
                yield
            else:
                with torch.autocast("cuda", enabled=autocast, dtype=torch.bfloat16):
                    yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            torch.backends.cudnn.benchmark = previous_benchmark


def precision_stage(matmul_tf32=True, autocast=None):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with inference_precision(matmul_tf32, autocast):
                return function(*args, **kwargs)
        return wrapped
    return decorate
