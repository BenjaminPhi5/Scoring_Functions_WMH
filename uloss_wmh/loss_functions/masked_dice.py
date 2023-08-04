

from monai.losses import MaskedDiceLoss

class MaskedDiceLossWrapperSingleClass():
    def __init__(self):
        self.base_loss = MaskedDiceLoss(include_background=False, sigmoid=True)
    def __call__(self, pred, target):
        pred = pred[:,1].unsqueeze(1) # get out just the WMH class for now for a fair comparison
        mask = target[:,-1].unsqueeze(1)
        target = target[:,1].unsqueeze(1)
        
        #print(pred.shape, target.shape)
        return self.base_loss(pred, target, mask)