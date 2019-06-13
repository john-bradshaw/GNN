
import enum
import typing

import numpy as np
import torch
from torch.nn import functional as F


class NdTensor(enum.Enum):
    """
    How tensor data is represented
    eg whether data is torch Tensor or Numpy array
    """
    NUMPY = 'numpy'
    TORCH = 'torch'


def work_out_nd_or_tensor(var) -> NdTensor:
    if isinstance(var, torch.Tensor):
        return NdTensor.TORCH
    elif isinstance(var, np.ndarray):
        return NdTensor.NUMPY
    else:
        raise RuntimeError


Nd_Ten = typing.Union[np.ndarray, torch.Tensor]
Op_Nd_Ten = typing.Union[np.ndarray, torch.Tensor, None]


def concatenate(variables: typing.List[Nd_Ten], axis=0) -> Nd_Ten:
    variant = work_out_nd_or_tensor(variables[0])
    if variant is NdTensor.NUMPY:
        return np.concatenate(variables, axis=axis)
    else:
        return torch.cat(variables, dim=axis)


def concatenate_maintaining_uniqueness(variables: typing.List[Nd_Ten], axis=0, inplace=False) -> Nd_Ten:
    """
    Variant where we ensure that all the elements of later items in the are shifted up so that they maintain equal value
    """
    prev_max = [0]

    def mapping_func(var):

        if inplace:
            var += prev_max[0]
        else:
            var = var + prev_max[0]

        prev_max[0] = var.max()
        return var

    variables = list(map(mapping_func, variables))
    return concatenate(variables, axis=axis)


def pad_right_2d(var: Nd_Ten, pad_right_amnt: int) -> Nd_Ten:
    variant = work_out_nd_or_tensor(var)
    if variant is NdTensor.NUMPY:
        return np.pad(var, ((0, 0), (0, pad_right_amnt)), mode='constant', constant_values=0)
    else:
        return F.pad(var, (0, pad_right_amnt, 0, 0), mode='constant', value=0)


def pad_bottom_2d(var: Nd_Ten, pad_bottom_amnt: int) -> Nd_Ten:
    variant = work_out_nd_or_tensor(var)
    if variant is NdTensor.NUMPY:
        return np.pad(var, ((0, pad_bottom_amnt), (0, 0)), mode='constant', constant_values=0)
    else:
        return F.pad(var, (0, 0, 0, pad_bottom_amnt), mode='constant', value=0)





def cumsum(var: Nd_Ten) -> Nd_Ten:
    variant = work_out_nd_or_tensor(var)
    if variant is NdTensor.NUMPY:
        return np.cumsum(var, 0)
    else:
        return torch.cumsum(var, 0)


def add_zero_at_start_of_one_d(var: Nd_Ten) -> Nd_Ten:
    variant = work_out_nd_or_tensor(var)
    if variant is NdTensor.NUMPY:
        start = np.array([0], dtype=var.dtype)
    else:
        start = torch.tensor([0], dtype=var.dtype, device=str(var.device))
    return concatenate([start, var])

