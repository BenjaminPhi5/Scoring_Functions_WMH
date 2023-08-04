import torch
import warnings

def normalize_inputs(input, sigmoid=False, softmax=False):
    if sigmoid:
        input = torch.sigmoid(input)
    channels = input.shape[1]
    if softmax:
        if channels == 1:
            warnings.warn("single channel prediction, `softmax=True` ignored.")
        else:
            input = torch.softmax(input, dim=1)

    return input