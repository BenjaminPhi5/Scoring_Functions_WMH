"""
Wrapped cross entropy, we do sum and then mean 
"""
from  torch.nn.functional import cross_entropy
from torch.nn import Module

class CrossEntropyWrapper(Module):
    def __init__(self, weight=None, label_smoothing=0.0, reduction='mean_sum'):
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, predictions, targets):
        if self.reduction == "mean":
            return cross_entropy(predictions, targets, weight=self.weight, label_smoothing=self.label_smoothing, reduction='mean')
        elif self.reduction == "sum":
            return cross_entropy(predictions, targets, weight=self.weight, label_smoothing=self.label_smoothing, reduction='sum')
        elif self.reduction == "mean_sum":
            dims = len(predictions.shape)
            return cross_entropy(
                predictions, targets, weight=self.weight, label_smoothing=self.label_smoothing,
                reduction='none'
            ).sum(dims=[i for i in range(2, dims)]).mean(dim=1).mean(dim=0)
