import utils.transforms as transforms
import numpy as np


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, mask_paths, dim, challenge, norm=False, sum_of_squares=False):
        """
        Args:
            mask_paths (str): Path to undersampling masks
            dim (int, int): Input dimension
            challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            norm (bool): Normalize either true or false
            sum_of_squares (bool): Compute the Root Sum of Squares (RSS) either true or false
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')

        self.dim = dim
        self.challenge = challenge
        self.norm = norm
        self.norm_shape = np.sqrt(dim[0]*dim[1])
        self.undermasks = {mask_path.split("218x")[-1].split(".")[0]: np.load(mask_path) for mask_path in mask_paths}
        self.sum_of_squares = sum_of_squares

    def ii(self, kspace, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (rows, cols, 24) for multi-coil
                data or (rows, cols, 2) for single coil data.
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Undersampled input image converted to a torch Tensor
                target (torch.Tensor): Target image converted to a torch Tensor.
                attrs (dict): Acquisition related information stored in the HDF5 object.
                fname (str): File name
                slice (int): Serial number of the slice.
        """
        under_masks = self.undermasks[str(kspace.shape[1])][np.random.randint(0, 100)]

        target_kspace = kspace.copy()
        kspace[~under_masks, :] = 0

        if self.norm:
            kspace = kspace / self.norm_shape
            target_kspace = target_kspace / self.norm_shape
            #kspace, norm = normalize(kspace)
            #target_kspace = target_kspace / norm

        target_image = np.empty(np.shape(target_kspace))
        image = np.fft.ifft2(target_kspace[:, :, ::2]+1j*target_kspace[:, :, 1::2], axes=(0, 1))
        target_image[:, :, ::2] = image.real
        target_image[:, :, 1::2] = image.imag

        masked_image = np.empty(np.shape(kspace))
        image2 = np.fft.ifft2(kspace[:, :, ::2]+1j*kspace[:, :, 1::2], axes=(0, 1))
        masked_image[:, :, ::2] = image2.real
        masked_image[:, :, 1::2] = image2.imag

        if self.challenge == 'multicoil' and self.sum_of_squares:
            masked_image = transforms.sum_of_squares(masked_image)
            target_image = transforms.sum_of_squares(target_image)

        masked_image = masked_image.transpose(2,0,1)
        target_image = target_image.transpose(2,0,1)
        return masked_image, target_image, under_masks, attrs, fname, slice

    def ki(self, kspace, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (rows, cols, 24) for multi-coil
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Undersampled input kspace converted to a torch Tensor
                target (torch.Tensor): Target image converted to a torch Tensor.
                attrs (dict): Acquisition related information stored in the HDF5 object.
                fname (str): File name
                slice (int): Serial number of the slice.
        """

        under_masks = self.undermasks[str(kspace.shape[1])][np.random.randint(0, 100)]

        target_kspace = kspace.copy()
        
        kspace[~under_masks, :] = 0
        if self.norm:
            kspace = kspace / self.norm_shape
            target_kspace = target_kspace / self.norm_shape
            #kspace, norm = normalize(kspace)
            #target_kspace = target_kspace / norm

        target_image = np.empty(np.shape(target_kspace))
        image = np.fft.ifft2(target_kspace[:, :, ::2]+1j*target_kspace[:, :, 1::2], axes=(0, 1))
        target_image[:, :, ::2] = image.real
        target_image[:, :, 1::2] = image.imag

        if self.challenge == 'multicoil' and self.sum_of_squares:
            kspace = transforms.sum_of_squares(kspace)
            target_image = transforms.sum_of_squares(target_image)
            
        kspace = kspace.transpose(2,0,1)
        target_image = target_image.transpose(2,0,1)
        return kspace, target_image, under_masks, attrs, fname, slice

    def ik(self, kspace, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (rows, cols, 24) for multi-coil
                data or (rows, cols, 2) for single coil data.
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Undersampled input image converted to a torch Tensor
                target (torch.Tensor): Target kspace converted to a torch Tensor.
                attrs (dict): Acquisition related information stored in the HDF5 object.
                fname (str): File name
                slice (int): Serial number of the slice.
        """

        under_masks = self.undermasks[str(kspace.shape[1])][np.random.randint(0, 100)]

        target_kspace = kspace.copy()
        kspace[~under_masks, :] = 0

        if self.norm:
            kspace = kspace / self.norm_shape
            target_kspace = target_kspace / self.norm_shape
            #kspace, norm = normalize(kspace)
            #target_kspace = target_kspace / norm

        masked_image = np.empty(np.shape(kspace))
        image = np.fft.ifft2(kspace[:, :, ::2]+1j*kspace[:, :, 1::2], axes=(0, 1))
        masked_image[:, :, ::2] = image.real
        masked_image[:, :, 1::2] = image.imag

        if self.challenge == 'multicoil' and self.sum_of_squares:
            masked_image = transforms.sum_of_squares(masked_image)
            target_kspace = transforms.sum_of_squares(target_kspace)

        masked_image = masked_image.transpose(2,0,1)
        target_kspace = target_kspace.transpose(2,0,1)
        return masked_image, target_kspace, under_masks, attrs, fname, slice

    def kk(self, kspace, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (rows, cols, 24) for multi-coil
                data or (rows, cols, 2) for single coil data.
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Undersampled input kspace converted to a torch Tensor
                target (torch.Tensor): Target kspace converted to a torch Tensor.
                attrs (dict): Acquisition related information stored in the HDF5 object.
                fname (str): File name
                slice (int): Serial number of the slice.
        """
        under_masks = self.undermasks[str(kspace.shape[1])][np.random.randint(0, 100)]

        target_kspace = kspace.copy()
        kspace[~under_masks, :] = 0
        if self.norm:
            kspace = kspace / self.norm_shape
            target_kspace = target_kspace / self.norm_shape
            #kspace, norm = normalize(kspace)
            #target_kspace = target_kspace / norm

        if self.challenge == 'multicoil' and self.sum_of_squares:
            kspace = transforms.sum_of_squares(kspace)
            target_kspace = transforms.sum_of_squares(target_kspace)
            
        kspace = kspace.transpose(2,0,1)
        target_kspace = target_kspace.transpose(2,0,1)
        return kspace, target_kspace, under_masks, attrs, fname, slice


def standardize(image):
    mean = image.mean(axis=(0, 1), keepdims=True)
    std = image.std(axis=(0, 1), keepdims=True)
    return (image - mean) / std


def normalize(image):
    #channel_min = image.reshape(-1, image.shape[-1]).min(dim=0).values
    #image = image - channel_min
    complex_abs = np.abs(image[:, :, ::2]+1j*image[:, :, 1::2])
    max_abs = np.max(complex_abs, axis=(0, 1, 2), keepdims=True)
    return image / max_abs, max_abs
