import torch
import math
import os
from torch.nn import functional as F
import glob
from tqdm import tqdm
from tqdm import trange
import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
from piq import VIFLoss
from piq import MultiScaleSSIMLoss
import sys

sys.path.append("..")
from models.unet.unet import WNet
from models.unet.unet import WNetDense
from models.unet.additional_layers import nrmse
import matplotlib.gridspec as gridspec

from torch.utils.tensorboard import SummaryWriter
from utils.utils import data_loaders, DotDict
from utils.ssim import ssim, SSIM, msssim, MSSSIM
from utils.metrics import metrics


# from configs import config_wnet


def train_net(params):
    # Initialize Parameters 
    params = DotDict(params)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    verbose = {}
    verbose['loss_train'], verbose['loss_valid'], verbose['psnr_train'], verbose['psnr_valid'], \
        verbose['ssim_train'], verbose['ssim_valid'], verbose['vif_train'], verbose['vif_valid'] = ([] for i in range(8))

    log_metrics = True
    ssim_module = SSIM()
    msssim_module = MSSSIM()
    vifLoss = VIFLoss(sigma_n_sq=0.4, data_range=1.)
    msssimLoss = MultiScaleSSIMLoss(data_range=1.)
    best_validation_metrics = 100

    train_generator, val_generator = data_loaders(params)
    loaders = {"train": train_generator, "valid": val_generator}

    wnet_identifier = params.mask_URate[0:2] + "WNet_dense=" + str(int(params.dense)) + "_" + params.architecture + "_" \
                      + params.lossFunction + '_lr=' + str(params.lr) + '_ep=' + str(params.num_epochs) + '_complex=' \
                      + str(int(params.complex_net)) + '_' + 'edgeModel=' + str(int(params.edge_model)) \
                      + '(' + str(params.num_edge_slices) + ')_date=' + (datetime.now()).strftime("%d-%m-%Y_%H-%M-%S")


    if not os.path.isdir(params.model_save_path):
        os.mkdir(params.model_save_path)
    print("\n\nModel will be saved at:\n", params.model_save_path)
    print("WNet ID: ", wnet_identifier)

    wnet, optimizer, best_validation_loss, preTrainedEpochs = generate_model(params, device)

    # data = (iter(train_generator)).next()

    # Adding writer for tensorboard. Also start tensorboard, which tries to access logs in the runs directory
    writer = init_tensorboard(iter(train_generator), wnet, wnet_identifier, device)

    for epoch in trange(preTrainedEpochs, params.num_epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                wnet.train()
            else:
                wnet.eval()

            for i, data in enumerate(loaders[phase]):

            # for i in range(10000):
                x, y_true, _, _, fname, slice_num = data
                x, y_true = x.to(device, dtype=torch.float), y_true.to(device, dtype=torch.float)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = wnet(x)
                    if params.lossFunction == 'mse':
                        loss = F.mse_loss(y_pred, y_true)
                    elif params.lossFunction == 'l1':
                        loss = F.l1_loss(y_pred, y_true)
                    elif params.lossFunction == 'ssim':
                        # standard SSIM
                        loss = 0.16 * F.l1_loss(y_pred, y_true) + 0.84 * (1 - ssim_module(y_pred, y_true))
                    elif params.lossFunction == 'msssim':
                        # loss = 0.16 * F.l1_loss(y_pred, y_true) + 0.84 * (1 - msssim_module(y_pred, y_true))
                        prediction_abs = torch.sqrt(torch.square(y_pred[:, 0::2]) + torch.square(y_pred[:, 1::2]))
                        target_abs = torch.sqrt(torch.square(y_true[:, 0::2]) + torch.square(y_true[:, 1::2]))
                        prediction_abs_flat = (torch.flatten(prediction_abs, start_dim=0, end_dim=1)).unsqueeze(1)
                        target_abs_flat = (torch.flatten(target_abs, start_dim=0, end_dim=1)).unsqueeze(1)
                        loss = msssimLoss(prediction_abs_flat, target_abs_flat)
                    elif params.lossFunction == 'vif':
                        prediction_abs = torch.sqrt(torch.square(y_pred[:, 0::2]) + torch.square(y_pred[:, 1::2]))
                        target_abs = torch.sqrt(torch.square(y_true[:, 0::2]) + torch.square(y_true[:, 1::2]))
                        prediction_abs_flat = (torch.flatten(prediction_abs, start_dim=0, end_dim=1)).unsqueeze(1)
                        target_abs_flat = (torch.flatten(target_abs, start_dim=0, end_dim=1)).unsqueeze(1)
                        loss = vifLoss(prediction_abs_flat, target_abs_flat)
                    elif params.lossFunction == 'mse+vif':
                        prediction_abs = torch.sqrt(torch.square(y_pred[:, 0::2]) + torch.square(y_pred[:, 1::2])).to(
                            device)
                        target_abs = torch.sqrt(torch.square(y_true[:, 0::2]) + torch.square(y_true[:, 1::2])).to(
                            device)
                        prediction_abs_flat = (torch.flatten(prediction_abs, start_dim=0, end_dim=1)).unsqueeze(1)
                        target_abs_flat = (torch.flatten(target_abs, start_dim=0, end_dim=1)).unsqueeze(1)
                        loss = 0.15 * F.mse_loss(prediction_abs_flat, target_abs_flat) + 0.85 * vifLoss(
                            prediction_abs_flat, target_abs_flat)
                    elif params.lossFunction == 'l1+vif':
                        prediction_abs = torch.sqrt(torch.square(y_pred[:, 0::2]) + torch.square(y_pred[:, 1::2])).to(
                            device)
                        target_abs = torch.sqrt(torch.square(y_true[:, 0::2]) + torch.square(y_true[:, 1::2])).to(
                            device)
                        prediction_abs_flat = (torch.flatten(prediction_abs, start_dim=0, end_dim=1)).unsqueeze(1)
                        target_abs_flat = (torch.flatten(target_abs, start_dim=0, end_dim=1)).unsqueeze(1)
                        loss = 0.146 * F.l1_loss(y_pred, y_true) + 0.854 * vifLoss(prediction_abs_flat, target_abs_flat)
                    elif params.lossFunction == 'msssim+vif':
                        prediction_abs = torch.sqrt(torch.square(y_pred[:, 0::2]) + torch.square(y_pred[:, 1::2])).to(
                            device)
                        target_abs = torch.sqrt(torch.square(y_true[:, 0::2]) + torch.square(y_true[:, 1::2])).to(
                            device)
                        prediction_abs_flat = (torch.flatten(prediction_abs, start_dim=0, end_dim=1)).unsqueeze(1)
                        target_abs_flat = (torch.flatten(target_abs, start_dim=0, end_dim=1)).unsqueeze(1)
                        loss = 0.66 * msssimLoss(prediction_abs_flat, target_abs_flat) + 0.33 * vifLoss(prediction_abs_flat, target_abs_flat)

                    if not math.isnan(loss.item()) and loss.item() < 2 * best_validation_loss:  # avoid nan/spike values
                        verbose['loss_' + phase].append(loss.item())
                        writer.add_scalar('Loss/' + phase + '_epoch_' + str(epoch), loss.item(), i)

                    if log_metrics and (
                            (i % params.verbose_gap == 0) or (phase == 'valid' and epoch > params.verbose_delay)):
                        y_true_copy = y_true.detach().cpu().numpy()
                        y_pred_copy = y_pred.detach().cpu().numpy()
                        y_true_copy = y_true_copy[:, ::2, :, :] + 1j * y_true_copy[:, 1::2, :, :]
                        y_pred_copy = y_pred_copy[:, ::2, :, :] + 1j * y_pred_copy[:, 1::2, :, :]
                        if params.architecture[-1] == 'k':
                            # transform kspace to image domain
                            y_true_copy = np.fft.ifft2(y_true_copy, axes=(2, 3))
                            y_pred_copy = np.fft.ifft2(y_pred_copy, axes=(2, 3))

                        # Sum of squares
                        sos_true = np.sqrt((np.abs(y_true_copy) ** 2).sum(axis=1))
                        sos_pred = np.sqrt((np.abs(y_pred_copy) ** 2).sum(axis=1))
                        '''
                        # Normalization according to: extract_challenge_metrics.ipynb
                        sos_true_max = sos_true.max(axis = (1,2),keepdims = True)
                        sos_true_org = sos_true/sos_true_max
                        sos_pred_org = sos_pred/sos_true_max
                        # Normalization by normalzing with ref with max_ref and rec with max_rec, respectively
                        sos_true_max = sos_true.max(axis = (1,2),keepdims = True)
                        sos_true_mod = sos_true/sos_true_max
                        sos_pred_max = sos_pred.max(axis = (1,2),keepdims = True)
                        sos_pred_mod = sos_pred/sos_pred_max
                        '''

                        '''
                        # normalization by mean and std
                        std = sos_pred.std(axis=(1, 2), keepdims=True)
                        mean = sos_pred.mean(axis=(1, 2), keepdims=True)
                        sos_pred_std = (sos_pred-mean) / std
                        std = sos_true.std(axis=(1, 2), keepdims=True)
                        mean = sos_pred.mean(axis=(1, 2), keepdims=True)
                        sos_true_std = (sos_true-mean) / std
                        '''
                        




                        '''
                        ssim, psnr, vif = metrics(sos_pred_org, sos_true_org)
                        ssim_mod, psnr_mod, vif_mod = metrics(sos_pred_mod, sos_true_mod)
                        '''
                        sos_true_max = sos_true.max(axis=(1, 2), keepdims=True)
                        sos_true_org = sos_true / sos_true_max
                        sos_pred_org = sos_pred / sos_true_max

                        ssim, psnr, vif = metrics(sos_pred, sos_true)
                        ssim_normed, psnr_normed, vif_normed = metrics(sos_pred_org, sos_true_org)

                        verbose['ssim_' + phase].append(np.mean(ssim_normed))
                        verbose['psnr_' + phase].append(np.mean(psnr_normed))
                        verbose['vif_' + phase].append(np.mean(vif_normed))

                        '''
                        print("===Normalization according to: extract_challenge_metrics.ipynb===")
                        print("SSIM: ", verbose['ssim_'+phase][-1])
                        print("PSNR: ", verbose['psnr_'+phase][-1])
                        print("VIF: ",  verbose['vif_' +phase][-1])
                        print("===Normalization by normalzing with ref with max_ref and rec with max_rec, respectively===")
                        print("SSIM_mod: ", np.mean(ssim_mod))
                        print("PSNR_mod: ", np.mean(psnr_mod))
                        print("VIF_mod: ",  np.mean(vif_mod))
                        print("===Normalization by dividing by the standard deviation of ref and rec, respectively===")
                        '''
                        print("Epoch: ", epoch)
                        print("SSIM: ", np.mean(ssim))
                        print("PSNR: ", np.mean(psnr))
                        print("VIF: ", np.mean(vif))

                        print("SSIM_normed: ", verbose['ssim_' + phase][-1])
                        print("PSNR_normed: ", verbose['psnr_' + phase][-1])
                        print("VIF_normed: ", verbose['vif_' + phase][-1])
                        '''
                        if True: #verbose['vif_' + phase][-1] < 0.4:
                            plt.figure(figsize=(9, 6), dpi=150)
                            gs1 = gridspec.GridSpec(3, 2)
                            gs1.update(wspace=0.002, hspace=0.1)
                            plt.subplot(gs1[0])
                            plt.imshow(sos_true[0], cmap="gray")
                            plt.axis("off")
                            plt.subplot(gs1[1])
                            plt.imshow(sos_pred[0], cmap="gray")
                            plt.axis("off")
                            plt.show()
                            # plt.pause(10)
                            # plt.close()
                        '''
                        writer.add_scalar('SSIM/' + phase + '_epoch_' + str(epoch), verbose['ssim_' + phase][-1], i)
                        writer.add_scalar('PSNR/' + phase + '_epoch_' + str(epoch), verbose['psnr_' + phase][-1], i)
                        writer.add_scalar('VIF/' + phase + '_epoch_' + str(epoch), verbose['vif_' + phase][-1], i)

                    print('Loss ' + phase + ': ', loss.item())

                    if phase == 'train':
                        if loss.item() < 2 * best_validation_loss:
                            loss.backward()
                            optimizer.step()


        # Calculate Averages
        psnr_mean = np.mean(verbose['psnr_valid'])
        ssim_mean = np.mean(verbose['ssim_valid'])
        vif_mean = np.mean(verbose['vif_valid'])
        validation_metrics = 0.2 * psnr_mean + 0.4 * ssim_mean + 0.4 * vif_mean

        valid_avg_loss_of_current_epoch = np.mean(verbose['loss_valid'])
        writer.add_scalar('AvgLoss/+train_epoch_' + str(epoch), np.mean(verbose['loss_train']), epoch)
        writer.add_scalar('AvgLoss/+valid_epoch_' + str(epoch), np.mean(verbose['loss_valid']), epoch)
        writer.add_scalar('AvgSSIM/+train_epoch_' + str(epoch), np.mean(verbose['ssim_train']), epoch)
        writer.add_scalar('AvgSSIM/+valid_epoch_' + str(epoch), ssim_mean, epoch)
        writer.add_scalar('AvgPSNR/+train_epoch_' + str(epoch), np.mean(verbose['psnr_train']), epoch)
        writer.add_scalar('AvgPSNR/+valid_epoch_' + str(epoch), psnr_mean, epoch)
        writer.add_scalar('AvgVIF/+train_epoch_' + str(epoch), np.mean(verbose['vif_train']), epoch)
        writer.add_scalar('AvgVIF/+valid_epoch_' + str(epoch), vif_mean, epoch)

        verbose['loss_train'], verbose['loss_valid'], verbose['psnr_train'], verbose['psnr_valid'], \
        verbose['ssim_train'], verbose['ssim_valid'], verbose['vif_train'], verbose['vif_valid'] = ([] for i in
                                                                                                    range(8))

        # Save Networks/Checkpoints
        if best_validation_metrics > validation_metrics:
            best_validation_metrics = validation_metrics
            best_validation_loss = valid_avg_loss_of_current_epoch
            save_checkpoint(wnet, params.model_save_path, wnet_identifier,
                            {'epoch': epoch + 1, 'state_dict': wnet.state_dict(),
                             'best_validation_loss': best_validation_loss, 'optimizer': optimizer.state_dict(), }, True)
        else:
            save_checkpoint(wnet, params.model_save_path, wnet_identifier,
                            {'epoch': epoch + 1, 'state_dict': wnet.state_dict(),
                             'best_validation_loss': best_validation_loss, 'optimizer': optimizer.state_dict(), },
                            False)


def save_checkpoint(model, model_save_path, name, state, is_best, filename='checkpoint.pth.tar'):
    print("=====Saving Network=====")
    if os.path.isdir(model_save_path):
        torch.save(state, model_save_path + 'BestVal=' + str(is_best) + '_' + name + '_' + filename)  # checkpoint
        torch.save(model, model_save_path + 'BestVal=' + str(is_best) + '_' + name + '.pth')  # model
    else:
        os.mkdir(model_save_path)
        torch.save(state, model_save_path + 'BestVal=' + str(is_best) + '_' + name + '_' + filename)
        torch.save(model, model_save_path + 'BestVal=' + str(is_best) + '_' + name + '.pth')


def generate_model(params, device):
    wnet = None
    preTrainedEpochs = 0
    best_validation_loss = 1e12
    if params['dense']:
        wnet = WNetDense(24, kspace_flag=True, architecture=params.architecture, mask_flag=params.mask_flag,
                         complex_flag=params.complex_net)
    else:
        wnet = WNet(24, kspace_flag=True, architecture=params.architecture, mask_flag=params.mask_flag,
                    complex_flag=params.complex_net)
        # always use kspace as input from dataloader. i.e., in config-file ki or kk, then set kspace_flag = True
        # as WNet will always transform image inputs to kspace (for data consistency) and then back to image, this saves one transformation step per slice 
    optimizer = torch.optim.Adam(wnet.parameters(), lr=params.lr)

    if params.resume is not None:
        if os.path.isfile(params.model_save_path + params.resume):
            checkpoint = torch.load(params.model_save_path + params.resume)
            wnet.load_state_dict(checkpoint['state_dict'])
            wnet.to(device)  # network must be on gpu before optimizer is loaded
            optimizer.load_state_dict(checkpoint['optimizer'])
            preTrainedEpochs = checkpoint['epoch']
            best_validation_loss = checkpoint['best_validation_loss']
            print("=> loaded checkpoint '{}' (epoch {})".format(params.resume, checkpoint['epoch']))
            print("Best ValidationLoss: ", checkpoint['best_validation_loss'])
        else:
            print("=> no checkpoint found at '{}'".format(params.model_save_path + params.resume))
            return
    else:
        wnet.to(device)
    best_validation_loss = 1e12
    return wnet, optimizer, best_validation_loss, preTrainedEpochs


def init_tensorboard(dataiter, wnet, wnet_identifier, device):
    # Adding writer for tensorboard. Also start tensorboard, which tries to access logs in the runs directory
    train, target, _, _, _, _ = dataiter.next()
    writer = SummaryWriter(log_dir='./runs/' + wnet_identifier)
    target = target.to(device, dtype=torch.float)
    writer.add_graph(wnet, (target))
    # %tensorboard --logdir=runs --port=8089
    return writer


if __name__ == "__main__":
    params = config_wnet.config
    print(params)
    train_net(params)
