import os
from warnings import warn
import numpy as np
import torch
import torch.nn as nn


def get_activation_by_name(activation):
    if activation == "leaky_relu":
        return nn.LeakyReLU
    elif activation == "relu":
        return nn.ReLU
    elif activation == "sigmoid":
        return nn.Sigmoid
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "softmax":
        return nn.Softmax
    elif activation == "identity":
        return nn.Identity
    else:
        return None


def get_optimizer_by_name(optimizer):
    if optimizer == "adam":
        return torch.optim.Adam
    elif optimizer == "adagrad":
        return torch.optim.Adagrad
    elif optimizer == "sgd":
        return torch.optim.SGD
    elif optimizer == "rmsprop":
        return torch.optim.RMSprop
    else:
        return torch.optim.Adam


def string_to_tuple(kernel_size):
    kernel_size_ = kernel_size.strip().replace('(', '').replace(')', '').split(',')
    kernel_size_ = [int(kernel) for kernel in kernel_size_]
    return kernel_size_


def string_to_list(int_list):
    int_list_ = int_list.strip().replace('[', '').replace(']', '').split(',')
    int_list_ = [int(v) for v in int_list_]
    return int_list_


def mkdir_if_not_exists(default_save_path):
    if not os.path.exists(default_save_path):
        os.mkdir(default_save_path)


def get_estimation(idx, target_name, estimation_dict):
    estimated = estimation_dict[target_name][idx]
    if len(estimated) == 0:
        warn('TODO: zero estimation, caused by ddp')
        return None
    estimated = np.concatenate([estimated[key] for key in sorted(estimated.keys())], axis=0)
    return estimated


def flat_word_set(word_set):
    return [subword for word in word_set for subword in word.split(' ')]
