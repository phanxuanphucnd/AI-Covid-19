# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import json
import torch
import matplotlib.pyplot as plt

from torchvision.transforms.functional import to_tensor

def load_json(file: str=None):
    """
    Function load a json file.

    Args:
        file: The file to load the object from.

    Returns:
        The loaded object.
    """
    with open(file) as json_file:
        data = json.load(json_file)

    return data

class AddGaussianNoise(object):
    def __init__(self, mean: int=0, std: float=5.) -> None:
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        if self.std == 0:
            return tensor
        
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(mean={0}, std={1}'.format(
            self.mean, self.std
        )

class ToFloatTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return to_tensor(pic).type(torch.FloatTensor)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class NoneTransform(object):
    def __call__(self, x):
        return x

def plot_roc_auc(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
            lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

def print_free_style(message, print_fun=print):
    print_fun("")
    print_fun("⫸  {} ".format(message))
    print_fun("")
