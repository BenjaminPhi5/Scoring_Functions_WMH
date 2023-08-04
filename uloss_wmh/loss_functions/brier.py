from torch.nn import Module
from  uloss_wmh.loss_functions.utils import normalize_inputs
from torch.nn.functional import mse_loss
import torch
from monai.losses import DiceLoss

class Brier(Module):
    def __init__(self, include_background=False, sigmoid=False, softmax=False, weight=1, reduce=True):
        super().__init__()
        self.include_background=include_background
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.weight = weight
        self.reduce = reduce

    def forward(self, predictions, targets):
        predictions = normalize_inputs(predictions, self.sigmoid, self.softmax)

        if not self.include_background and predictions.shape[1] != 1:
            predictions = predictions[:,1:]
            targets = targets[:,1:]
            
        sum_dims = torch.arange(2, len(predictions.shape), 1).tolist()
        
        if self.reduce:
            return self.weight * mse_loss(predictions, targets.type(torch.float32), reduction='none').sum(dim=sum_dims).mean()
        
        else:
            return self.weight * mse_loss(predictions, targets.type(torch.float32), reduction='none').sum(dim=1) / predictions.shape[1]
    
class BrierPlusDice(Module):
    def __init__(self, include_background=False, sigmoid=False, softmax=False, brier_factor=1., dice_factor=1.):
        super().__init__()
        self.brier =Brier(include_background, sigmoid, softmax, 1.)
        self.dice = DiceLoss(sigmoid=sigmoid, softmax=softmax, include_background=include_background)
        self.brier_factor = brier_factor
        self.dice_factor = dice_factor
        self.global_factor = 1/(brier_factor + dice_factor)
        
    def forward(self, input, target):
        return (self.brier_factor * self.global_factor) * self.brier(input, target) + (self.dice_factor * self.global_factor) * self.dice(input, target)
        # return self.dice(input, target)