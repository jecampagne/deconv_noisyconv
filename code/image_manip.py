import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np


############################ Convolution utils ################################
def make_psf(fwhm):
    """model de PSF"""
    # The FWHM of a Gaussian is 2 sqrt(2 ln2) sigma
    _fwhm_factor = 2.3548200450309493

    assert fwhm >= 0.0, "FATAL: fwhm must be >=0"
    fwhm = max(fwhm, 1e-16)  # JEC 26May25 allow call fwhm=0

    sigma = fwhm / _fwhm_factor
    _inv_sigsq = 1.0 / sigma**2

    # JEC 26May25: add 1 bin to get 0,0 centered
    W = np.linspace(-5.0 * sigma, 5.0 * sigma, int(10 * sigma) + 1, endpoint=True)
    H = np.linspace(-5.0 * sigma, 5.0 * sigma, int(10 * sigma) + 1, endpoint=True)

    X, Y = np.meshgrid(W, H)
    rsq = X**2 + Y**2

    y = np.exp(-0.5 * rsq * _inv_sigsq)

    # normalization to 1
    y /= y.sum()

    return y


def make_psf_old(fwhm):
    """model de PSF"""
    # The FWHM of a Gaussian is 2 sqrt(2 ln2) sigma
    _fwhm_factor = 2.3548200450309493

    sigma = fwhm / _fwhm_factor
    _inv_sigsq = 1.0 / sigma**2
    W = np.linspace(-5 * sigma, 5 * sigma, int(10 * sigma), endpoint=True)
    H = np.linspace(-5 * sigma, 5 * sigma, int(10 * sigma), endpoint=True)

    X, Y = np.meshgrid(W, H)
    rsq = X**2 + Y**2

    y = np.exp(-0.5 * rsq * _inv_sigsq)

    # normalization to 1
    y /= y.sum()

    return y


def rescale_image_range(im, max_I=1.0, min_I=-1.0):
    temp = (im - im.min()) / ((im.max() - im.min()))
    return temp * (max_I - min_I) + min_I


############################ Data Augmentation utils ##########################


class ToTensor(object):
    """Transform the tensor from HWC(TensorFlow default) to CHW (Torch)"""

    def __call__(self, pic):
        # use copy here to avoid crash
        img = torch.from_numpy(pic.transpose((2, 0, 1)).copy())
        return img

    def __repr__(self):
        return self.__class__.__name__ + "()"


class RandomApply(transforms.RandomApply):
    """Apply randomly a list of transformations with a given probability

    Args:
        transforms (list or tuple): list of transformations
        p (float): list of probabilities
    """

    def __init__(self, transforms):
        super(RandomApply, self).__init__(transforms)

    def __call__(self, img):
        # for each list of transforms
        # apply random sample to apply or not the transform
        for itset in range(len(self.transforms)):
            transf = self.transforms[itset]
            t = random.choice(transf)
            #### print('t:=',t)
            img = t(img)

        return img


def flipH(a):
    """
    Parameters
    ----------
    a: an image

    Returns
    -------
    an image flipped wrt the Horizontal axe
    """
    return np.flip(a, 0)


def flipV(a):
    """
    Parameters
    ----------
    a: an image

    Returns
    -------
    an image flipped wrt the Vertical axe
    """
    return np.flip(a, 1)


def rot90(a):
    """
    Parameters
    ----------
    a: an image

    Returns
    -------
    an image rotated 90deg anti-clockwise
    """
    return np.rot90(a, 1)


def rot180(a):
    """
    Parameters
    ----------
    a: an image

    Returns
    -------
    an image rotated 180deg anti-clockwise
    """
    return np.rot90(a, 2)


def rot270(a):
    """
    Parameters
    ----------
    a: an image

    Returns
    -------
    an image rotated 270deg anti-clockwise
    """
    return np.rot90(a, 3)


def identity(a):
    """
    Parameters
    ----------
    a: an image

    Returns
    -------
    the same image
    """
    return a
