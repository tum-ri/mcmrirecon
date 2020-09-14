import torch
import numpy as np
import matplotlib.pylab as plt
import os
import h5py
from utils.utils import test_data_loader, data_loaders, DotDict
from configs import config_wnet
from models.unet.unet import WNetDense, WNet


def run(params, net, val=False):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    wnet = None
    if net.split("pth")[1] == ".tar":
        if params.dense:
            wnet = WNetDense(24, kspace_flag=True, architecture=params.architecture, mask_flag=params.mask_flag,
                             complex_flag=params.complex_net)
        else:
            wnet = WNet(24, kspace_flag=True, architecture=params.architecture, mask_flag=params.mask_flag,
                        complex_flag=params.complex_net)

        checkpoint = torch.load(net)
        wnet.load_state_dict(checkpoint['state_dict'])
        wnet.to(device)
        wnet.eval()
    else:
        wnet = torch.load(net)
        wnet.to(device)
        wnet.eval()

    if val:
        _, test_generator = data_loaders(params)  # val loader
    else:
        test_generator = test_data_loader(params)  # test loader

    t1_r5_path = "./data/Track01/12-channel-R=5/"

    try:
        os.makedirs(t1_r5_path)
    except:
        pass

    slice_idx = 120
    save_slice = True
    tmp = torch.Tensor()
    gt = torch.Tensor()
    for i, data in enumerate(test_generator, 1):
        if val:
            x, y_true, _, _, fname, slice = data
            x = x.cuda()
            if (slice_idx + params['slice_cut_val'][0]) % slice[0].numpy() == 0:
                gt = np.sqrt((np.abs(y_true[0]) ** 2).sum(axis=0))
        else:
            x, fname, slice = data
            b, h, w, c = x.shape
            x = x / np.sqrt(h * w)
            x = x.permute(0, 3, 1, 2).cuda()

        y = wnet(x)

        y = y.permute(0, 2, 3, 1).detach().cpu().numpy()
        b, h, w, c = y.shape
        if params.architecture[-1] == 'k':
            y = y * np.sqrt(h * w)
            y = np.fft.ifft2(y[:, :, :, ::2]+1j*y[:, :, :, 1::2], axes=(1, 2))
        else:
            y = np.fft.fft2(y[:, :, :, ::2]+1j*y[:, :, :, 1::2], axes=(1, 2))
            y = y*np.sqrt(h*w)
            y = np.fft.ifft2(y, axes=(1, 2))

        y = np.sqrt((np.abs(y) ** 2).sum(axis=3))
        tmp = torch.cat((tmp, torch.from_numpy(y).float()))
        print("saved slice: " + str(slice[0].numpy()) + " to file: " + str(fname[0]))

        if i % 156 == 0 and i != 0:
            with h5py.File(t1_r5_path + str(fname[0]), 'w') as f:
                f.create_dataset('reconstruction', data=tmp)
                test_image = f['reconstruction'][slice_idx]
                if save_slice and val:
                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
                    axes[0].imshow(gt.numpy(), cmap = "gray")
                    axes[0].axis('off')
                    axes[1].imshow(test_image, cmap = "gray")
                    axes[1].axis('off')
                    plt.savefig('img' + str(slice_idx) + str(fname[0]) + '.png', dpi=100)
                    print('saved slice ' + str(slice_idx) + ' to image')
                elif save_slice:
                    plt.imshow(test_image, cmap = "gray")
                    plt.axis('off')
                    plt.savefig('img' + str(slice_idx) + str(fname[0]) + '.png', dpi=100)
                    print('saved slice ' + str(slice_idx) + ' to image')

                f.close()
                print('saved reconstruction to ' + str(fname[0]))
                tmp = torch.Tensor()


if __name__ == "__main__":
    params = config_wnet.config
    params['batch_size'] = 1
    params['slice_cut_val'] = (50, 50)
    params['norm'] = True
    params['architecture'] = 'iiiiii'
    net = r"..\BestVal=True_R5WNet_dense=1_iiiiii_msssim+vif_lr=0.0005_ep=50_complex=1_edgeModel=0(0)_date=24-07-2020_13-28-38.pth"
    run(DotDict(params), net, val=False) # set val to True to use Val Set
