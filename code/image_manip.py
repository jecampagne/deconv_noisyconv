import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np


class ToTensor(object):
    """ Transform the tensor from HWC(TensorFlow default) to CHW (Torch)
    """
    def __call__(self, pic):
        # use copy here to avoid crash
        img = torch.from_numpy(pic.transpose((2, 0, 1)).copy())
        return img
    def __repr__(self):
        return self.__class__.__name__ + '()'


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
    return np.flip(a,0)

def flipV(a):
    """
    Parameters
    ----------
    a: an image

    Returns
    -------
    an image flipped wrt the Vertical axe
    """
    return np.flip(a,1)

def rot90(a):
    """
    Parameters
    ----------
    a: an image

    Returns
    -------
    an image rotated 90deg anti-clockwise
    """
    return np.rot90(a,1)

def rot180(a):
    """
    Parameters
    ----------
    a: an image

    Returns
    -------
    an image rotated 180deg anti-clockwise
    """
    return np.rot90(a,2)

def rot270(a):
    """
    Parameters
    ----------
    a: an image

    Returns
    -------
    an image rotated 270deg anti-clockwise
    """
    return np.rot90(a,3)

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
