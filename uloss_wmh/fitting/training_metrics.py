from monai.metrics import DiceMetric
import torch

class WrappedDiceMetric():
    def __init__(self):
        self.dm = DiceMetric(include_background=False)
        
    def __call__(self, pred, target):
        self.dm(torch.sigmoid(pred) > 0.5, target)
        return self.dm.aggregate().item()
    
class WrappedDiceMetricFrom2Channel():
    def __init__(self):
        self.dm = DiceMetric(include_background=False)
        
    def __call__(self, pred, target):
        self.dm(torch.softmax(pred, 1)[:,1].unsqueeze(1) > 0.5, target)
        return self.dm.aggregate().item()
    
class WrappedDiceMetricNTo1Channel():
    def __init__(self):
        self.dm = DiceMetric(include_background=False)
        
    def __call__(self, pred, target):
        self.dm(torch.softmax(pred, 1)[:,1].unsqueeze(1) > 0.5, target[:,0].unsqueeze(1))
        return self.dm.aggregate().item()