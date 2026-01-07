import random
from typing import Union
from pathlib import Path
import os
from device_manager import global_device

import numpy as np

import torch


def set_seeds(seed):
    if seed is None:
        return
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        
def set_devices(devices: str):
    """
    Set cuda devices

    Args:
        devices (str): Comma separated string of device numbers. e.g., "0", "0, 1"
    """
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)


def mkdir(directory: Union[str, Path]):
    if isinstance(directory, str):
        directory = Path(directory)

    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    return directory


def prepare_inputs(inputs):
    # move data to the current GPU
    if isinstance(inputs, torch.Tensor):
        return inputs.float().to(global_device)
    elif isinstance(inputs, (tuple, list)):
        return type(inputs)(prepare_inputs(v) for v in inputs)
