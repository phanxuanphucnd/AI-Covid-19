# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import json
import torch
import matplotlib.pyplot as plt

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
