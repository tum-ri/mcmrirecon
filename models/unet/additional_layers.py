import numpy as np

def nrmse(y_true, y_pred):
    """
    Normalized Root Mean Squared Error (NRMSE) - Euclidean distance normalization
    :param y_true: Reference
    :param y_pred: Predicted
    :return:
    """

    denom = np.max(y_true, axis=(1, 2, 3)) - np.min(y_true, axis=(1, 2, 3))
    return np.sqrt(np.mean(np.square(y_pred - y_true), axis=(1, 2, 3))) / denom


def nrmse_min_max(y_true, y_pred):
    """
     Normalized Root Mean Squared Error (NRMSE) - min-max normalization
     :param y_true: Reference
     :param y_pred: Predicted
     :return:
     """

    denom = np.sqrt(np.mean(np.square(y_true), axis=(1, 2, 3)))
    return np.sqrt(np.mean(np.square(y_pred - y_true), axis=(1, 2, 3))) / denom


def fft_layer(image):
    """
    Input: 2-channel array representing image domain complex data
    Output: 2-channel array representing k-space complex data
    """

    # get real and imaginary portions
    real = image[:, :, :, 0]
    imag = image[:, :, :, 1]

    image_complex = real + 1j*imag
    kspace_complex = np.fft.fft2(image_complex)

    return kspace_complex


def ifft_layer(kspace_2channel):
    """
    Input: 2-channel array representing k-space
    Output: 2-channel array representing image domain
    """
    # get real and imaginary portions
    real = kspace_2channel[:, :, :, 0]
    imag = kspace_2channel[:, :, :, 1]

    kspace_complex = real + 1j*imag
    image_complex = np.fft.ifft2(kspace_complex)

    return image_complex
