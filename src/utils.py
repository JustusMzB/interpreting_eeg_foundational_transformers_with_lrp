# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Copyright (c) 2024 Weibang Jiang
# MIT License (see LICENSE for details)
# Minor additions by Justus Meyer zu Bexten, denoted in code
# Based on BEiT-v2, timm, DeiT, DINO, and BIOT code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# https://github.com/ycq091044/BIOT
# ---------------------------------------------------------

import io
import os
import math
import time
import json
import glob
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict
from mne.channels import make_standard_montage
from pathlib import Path
import argparse

import torch
from torch import nn
import torch.distributed as dist
from torch import inf
import logging
import pickle
from scipy.signal import resample
from torchmetrics.classification import MulticlassAccuracy


LOGGER = logging.getLogger(__name__)

class BinaryBalancedAccuracy(MulticlassAccuracy):
    """
    Binary balanced accuracy metric, only for input with single logits.
    Necessary because torchmetrics 
    """
    def __init__(self, **kwargs):
        super().__init__(num_classes=2, average='macro', **kwargs)
    def update(self, preds, target):
        
        return super().update(torch.sigmoid(preds)>0.5, target)
    def forward(self, preds, target):
        if preds.ndim == 1:
            preds = preds.unsqueeze(1)
        return super().forward(torch.sigmoid(preds)>0.5, target)
standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

def find_tensors_with_grad(model: nn.Module):
    # By Justus Meyer zu Bexten
    """
    Recursively finds all tensors in a model that require gradients.

    Args:
        model (nn.Module): The PyTorch model to search.

    Returns:
        A dictionary where keys are the paths to the tensors and
        values are the tensors themselves.
    """
    tensors_with_grad = {}
    for name, module in model.named_modules():
        # Check attributes of the module
        for attr_name, attr in module.__dict__.items():
            if isinstance(attr, torch.Tensor) and attr.requires_grad:
                full_name = f"{name}.{attr_name}" if name else attr_name
                tensors_with_grad[full_name] = attr
        
        # Check parameters explicitly (covers cases where they aren't direct attributes)
        for param_name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                full_name = f"{name}.{param_name}" if name else param_name
                tensors_with_grad[full_name] = param
                
    return tensors_with_grad

def adapt_channel_names_to_mne(ch_names):
    """
    Attempts to adapt a list of channel names to the MNE standard, with regards to case etc.
    """
    # By Justus Meyer zu Bexten
    standard_1020_according_to_mne = make_standard_montage('standard_1020').ch_names
    adapted = []
    for ch in ch_names:
        for std_ch in standard_1020_according_to_mne:
            if ch.lower().replace(".", '') == std_ch.lower():
                adapted.append(std_ch)
                break
        else:
            adapted.append(ch)
    return adapted
    

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model
            
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        LOGGER.debug("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        LOGGER.debug("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        LOGGER.debug("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def get_input_chans(ch_names):
    input_chans = [0] # for cls token
    for ch_name in ch_names:
        input_chans.append(standard_1020.index(ch_name.upper().replace('.', '')) + 1)
    return input_chans

def get_chan_names(ch_indices):
    ch_names = []
    for ch_index in ch_indices:
        if ch_index == 0:
            LOGGER.debug("skpping cls token...")
        else:
            ch_names.append(standard_1020[ch_index - 1])
    return ch_names

def normalize_electrode_name(name: str) -> str:
    """Normalize an electrode name for robust string matching."""
    cleaned = name.upper().strip().replace('.', '').replace('-', '').replace(' ', '')
    alias_map = {
        'FFC6': 'FC6',
        'FFC5': 'FC5',
    }
    return alias_map.get(cleaned, cleaned)

def parse_electrode_list(electrodes_arg):
    """
    Parses a string (e.g. 'P6,FCz,FC2' or 'AF7 FP1 FPZ') or iterable of strings
    into a set of normalized electrode names.
    """
    if not electrodes_arg:
        return set()
    if isinstance(electrodes_arg, str):
        import re
        tokens = [t for t in re.split(r'[, \t\n]+', electrodes_arg) if t]
    else:
        tokens = list(electrodes_arg)
    return {normalize_electrode_name(t) for t in tokens}

def filter_eeg_channels(ch_names, exclude_electrodes=None, include_electrodes=None):
    """
    Given a list of dataset channel names `ch_names`, returns:
      keep_indices: List[int] of 0-based channel indices in `ch_names` to keep
      excluded_indices: List[int] of 0-based channel indices in `ch_names` excluded
      filtered_ch_names: List[str] of kept channel names
    """
    exclude_set = parse_electrode_list(exclude_electrodes) if exclude_electrodes else set()
    include_set = parse_electrode_list(include_electrodes) if include_electrodes else set()

    keep_indices = []
    excluded_indices = []
    filtered_ch_names = []

    for idx, ch in enumerate(ch_names):
        norm_ch = normalize_electrode_name(ch)
        is_included = (norm_ch in include_set) if include_set else True
        is_excluded = (norm_ch in exclude_set)

        if is_included and not is_excluded:
            keep_indices.append(idx)
            filtered_ch_names.append(ch)
        else:
            excluded_indices.append(idx)

    return keep_indices, excluded_indices, filtered_ch_names



class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y
    

class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y
    

def prepare_TUEV_dataset(root):
    # set random seed
    seed = 4523
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "processed_train"))
    val_files = os.listdir(os.path.join(root, "processed_eval"))
    test_files = os.listdir(os.path.join(root, "processed_test"))

    # prepare training and test data loader
    train_dataset = TUEVLoader(
        os.path.join(
            root, "processed_train"), train_files
    )
    test_dataset = TUEVLoader(
        os.path.join(
            root, "processed_test"), test_files
    )
    val_dataset = TUEVLoader(
        os.path.join(
            root, "processed_eval"), val_files
    )
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_TUAB_dataset(root):
    # set random seed
    seed = 12345
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = TUABLoader(os.path.join(root, "train"), train_files)
    test_dataset = TUABLoader(os.path.join(root, "test"), test_files)
    val_dataset = TUABLoader(os.path.join(root, "val"), val_files)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset

