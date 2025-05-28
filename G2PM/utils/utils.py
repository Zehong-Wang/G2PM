import os
import os.path as osp
import random
from pathlib import Path
import numpy as np
import torch
import psutil


def get_memory_usage():
    """Get both CPU and GPU memory usage."""
    # Get CPU memory usage
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / (1024 ** 2)

    # Get GPU memory usage (if available)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
    else:
        gpu_memory = 0

    return cpu_memory, gpu_memory


def get_device_from_model(model):
    return next(model.parameters()).device


def check_path(path):
    if not osp.exists(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_millions(num):
    return round(num / 1e6, 2)


def mask2idx(mask):
    return torch.where(mask == True)[0]


def idx2mask(idx, num_instances):
    mask = torch.zeros(num_instances, dtype=torch.bool)
    mask[idx] = 1
    return mask
