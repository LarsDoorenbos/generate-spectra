
from typing import Union
import os
import shutil

import numpy as np

from torch import nn
from torchvision.transforms.functional import normalize

ParallelType = Union[nn.DataParallel, nn.parallel.DistributedDataParallel]


class WithStateDict(nn.Module):
    """Wrapper to provide a `state_dict` method to a single tensor."""
    def __init__(self, **tensors):
        super().__init__()
        for name, value in tensors.items():
            self.register_buffer(name, value)
        # self.tensor = nn.Parameter(tensor, requires_grad=False)


def expanduservars(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def archive_code(path: str, method: str) -> None:
    shutil.copy("params_" + method + ".yml", path)
    # Copy the current code to the output folder.
    os.system(f"git ls-files -z | xargs -0 tar -czf {os.path.join(path, 'code.tar.gz')}")


def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean / std
    _std = 1 / std
    return normalize(tensor, _mean, _std)


class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean / std
        self._std = 1 / std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1, 1, 1)) / self._std.reshape(-1, 1, 1)
        return normalize(tensor, self._mean, self._std)