import torch.nn as nn
from enum import Enum
from uloss_wmh.loss_functions.utils import normalize_inputs
from torch.nn.functional import mse_loss, l1_loss

"""
The issue with AVD is that I need a differentiable loss.
I could modify the weight by the confidence, or take the pixels above a certain threshold
and only apply the loss at those places....
"""

class VolumeDifferenceMethod(Enum):
    SUM = 1
    THRESHOLD = 2
    WEIGHTED = 3
    WEIGHTED_THRESHOLDED = 4

class VolumeDifferenceLoss(nn.Module):
    def __init__(self, method: VolumeDifferenceMethod, threshold=0.5, weight=1., sigmoid=False, softmax=True, l2=False):
        super().__init__(self)
        self.method = method
        self.threshold = threshold
        self.weight = weight
        self.softmax = softmax
        self.sigmoid = sigmoid
        self.l2 = False

    def forward(self, prediction, target):
        """
        prediction and target should be the shape [B,C,<dims>], C is number of classes
        """
        prediction = normalize_inputs(prediction, self.sigmoid, self.softmax)
        num_dims = len(prediction.shape)
        spatial_dims = [d for d in range(2, num_dims)]

        if self.method == VolumeDifferenceMethod.SUM:
            prediction_sum = prediction.sum(dims=spatial_dims)
            target_sum = target.sum(dims=spatial_dims)
            
        elif self.method == VolumeDifferenceMethod.THREHOLD:
            prediction_sum = prediction[prediction > self.threshold].sum(dims=spatial_dims)
            target_sum = target.sum(dims=spatial_dims)
            
        elif self.method == VolumeDifferenceMethod.WEIGHTED:
            prediction_sum = prediction.sum(dims=spatial_dims)
            target_sum = (target * prediction).sum(dims=spatial_dims)

        elif self.method == VolumeDifferenceMethod.WEIGHTED_THRESHOLDED:
            prediction_sum = prediction[prediction > self.threshold].sum(dims=spatial_dims)
            target_sum = (target * prediction).sum(dims=spatial_dims)
            
        else:
            raise ValueError("method not recongnised. One of VolumeDifferenceMethod[SUM, THRESHOLD, WEIGHTED, WEIGHTED_THRESHOLDED] is accepted")

        if self.l2:
            return mse_loss(prediction_sum, target_sum, reduction=None).mean(dim=1).mean(dim=0)
        else:
            return l1_loss(prediction_sum, target_sum, reduction=None).mean(dim=1).mean(dim=0)
            