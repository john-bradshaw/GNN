

from os import path
from pathlib import Path

import shutil
import os
import pickle
import configparser

import arrow
import torch


ENV_NAME = 'GNN'

#todo: maybe a double dict useful: http://code.activestate.com/recipes/578224/


def get_qm9_data_path():
    home = str(Path.home())

    config_file = path.join(home, '.host_experiment_settings.ini')
    config = configparser.ConfigParser()
    config.read(config_file)

    return config['GNN']['qm9_data_path']

def find_inverse_dict(dict_in):
    return {value:key for key, value in dict_in.items()}


def reindex_based_on_idx_map(old_indices, indices_map):
    return [indices_map[i] for i in old_indices]


class CudaDetails(object):
    def __init__(self, use_cuda:bool, gpu_id=None):
        self.use_cuda = use_cuda
        self.gpu_id = gpu_id

    def return_cudafied(self, arg):
        if self.use_cuda:
            arg = arg.cuda(self.gpu_id)
        return arg

    @property
    def r_mod(self):
        if self.use_cuda:
            return torch.cuda
        else:
            return torch

    @property
    def device_str(self):
        if self.use_cuda:
            return 'cuda:0'
        else:
            return 'cpu'


def from_np_to_cuda(numpy_array, cuda_details: CudaDetails):
    return cuda_details.return_cudafied(torch.from_numpy(numpy_array))


def save_checkpoint(state, is_best, filename=None):
    name_prepender = str(os.getenv(ENV_NAME))

    if filename is None:
        date_str = arrow.now().format('HH-mm_Do_MMMM')
        filename = 'checkpoint_{}.pth.pick'.format(date_str)

    if name_prepender is not None:
        filename = name_prepender + filename

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.pick')


def to_pickle(obj_to_dump, filename):
    with open(filename, 'wb') as fo:
        pickle.dump(obj_to_dump, fo)


def from_pickle(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo)
    return data


class AverageMeter(object):
    """Computes and stores the average and current value

    taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L95-L113"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
