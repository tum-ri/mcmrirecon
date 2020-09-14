import scipy.fftpack
import numpy as np
import skimage.measure
import scipy
import math
import torch
from glob import glob


from pathlib import Path
from torch.utils.data import DataLoader
from data_handlers.dataloader import SliceData
from data_handlers.datatransformer import DataTransform
from .transforms import sum_of_squares

def SSIM(x_good, x_bad):
    assert len(x_good.shape) == 2
    ssim_res = skimage.metrics.structural_similarity(x_good, x_bad)
    return ssim_res


def PSNR(x_good, x_bad):
    assert len(x_good.shape) == 2
    psnr_res = skimage.metrics.peak_signal_noise_ratio(x_good, x_bad)
    return psnr_res


def NMSE(x_good, x_bad):
    assert len(x_good.shape) == 2
    nmse_a_0_1 = np.sum((x_good - x_bad) ** 2)
    nmse_b_0_1 = np.sum(x_good ** 2)
    # this is DAGAN implementation, which is wrong
    nmse_a_0_1, nmse_b_0_1 = np.sqrt(nmse_a_0_1), np.sqrt(nmse_b_0_1)
    nmse_0_1 = nmse_a_0_1 / nmse_b_0_1
    return nmse_0_1


def computePSNR(o_, p_, i_):
    return PSNR(o_, p_), PSNR(o_, i_)


def computeSSIM(o_, p_, i_):
    return SSIM(o_, p_), SSIM(o_, i_)


def computeNMSE(o_, p_, i_):
    return NMSE(o_, p_), NMSE(o_, i_)


def adjust_learning_rate(optimizer, iters, base_lr, policy_parameter, policy='step', multiple=[1]):
    '''
    source: https://github.com/last-one/Pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/utils.py
    '''
    if policy == 'fixed':
        lr = base_lr
    elif policy == 'step':
        lr = base_lr * (policy_parameter['gamma'] ** (iters // policy_parameter['step_size']))
    elif policy == 'exp':
        lr = base_lr * (policy_parameter['gamma'] ** iters)
    elif policy == 'inv':
        lr = base_lr * ((1 + policy_parameter['gamma'] * iters) ** (-policy_parameter['power']))
    elif policy == 'multistep':
        lr = base_lr
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
            else:
                break
    elif policy == 'poly':
        lr = base_lr * ((1 - iters * 1.0 / policy_parameter['max_iter']) ** policy_parameter['power'])
    elif policy == 'sigmoid':
        lr = base_lr * (1.0 / (1 + math.exp(-policy_parameter['gamma'] * (iters - policy_parameter['stepsize']))))
    elif policy == 'multistep-poly':
        lr = base_lr
        stepstart = 0
        stepend = policy_parameter['max_iter']
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
                stepstart = stepvalue
            else:
                stepend = stepvalue
                break
        lr = max(lr * policy_parameter['gamma'], lr * (1 - (iters - stepstart) * 1.0 / (stepend - stepstart)) ** policy_parameter['power'])

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * multiple[i]
    return lr


def datasets(args, transform=None):
    """
    Creates training and validation dataset
    """
    mask_paths = glob(args.data_root + '/poisson_sampling/' + args.mask_URate + '*.npy')
    transform = DataTransform(mask_paths,
                              args.dim,
                              args.challenge,
                              args.norm,
                              args.sum_of_squares) if transform is None else transform

    # Generating Datasets.
    train_dataset = SliceData(
        root=Path(args.data_root) / f'Train',
        data_set="train",
        dim=args.dim,
        domain=args.domain,
        transform=transform,
        challenge=args.challenge,
        sample_rate=args.sample_rate,
        slice_cut=args.slice_cut_train,
        num_edge_slices=args.num_edge_slices,
        edge_model=args.edge_model,
    )

    val_dataset = SliceData(
        root=Path(args.data_root) / f'Val',
        data_set="val",
        dim=args.dim,
        domain=args.domain,
        transform=transform,
        challenge=args.challenge,
        sample_rate=args.sample_rate,
        slice_cut=args.slice_cut_val,
        num_edge_slices=args.num_edge_slices,
        edge_model=args.edge_model,
    )
    return train_dataset, val_dataset

def testset(args):
    """
    Creates test dataset
    """
    # Generating Datasets.
    test_dataset = SliceData(
        root=Path(args.data_root + '/test_12_channel/Test-R=' + args.mask_URate.split("R")[1]),
        data_set="test",
        dim=args.dim,
        domain=args.domain,
        transform=None,
        challenge=args.challenge,
        sample_rate=args.sample_rate,
        slice_cut=args.slice_cut_val,
    )

    return test_dataset

def data_loaders(args, transform=None):
    """
    Creates training and validation data loader
    """
    train_dataset, val_dataset = datasets(args, transform)

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader

def test_data_loader(args):
    """
    Creates test data loader
    """
    test_dataset = testset(args)

    # Generating Data Loaders
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    return test_loader

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

