import os
import torch
import numpy as np
from typing import Union


def mkdirs(paths: Union[list, str]):
    """create empty directories if they don't exist

    :param paths: a list of directory paths or a single path
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path: str):
    """create a single empty directory if it didn't exist

    :param path: a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def tensor2im(input_image: Union[torch.Tensor, np.ndarray], imtype=np.uint8) -> np.array:
    """Converts a Tensor array into a numpy image array.

    :param input_image: the input image tensor array
    :param imtype: the desired type of the converted numpy array
    :return: Converted Image
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        # post-processing: transpose and scaling
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    """
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    # tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
