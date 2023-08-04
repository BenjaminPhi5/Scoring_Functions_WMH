# I want to have the different versions where I decide how to apply
# the batching.

# also, I should try Dice++ as well, which is shown to improve calibration... I ideally want to be able to combine them in various ways...

"""
For now, I think that I will use MONAI's built in functions and modify them if I think that is useful
e.g to add a masking wrapper for example.

Lets just test which of the MONAI segmentation losses works best, and then build up the topk loss and other losses etc.

Ideally, I will build a single class that allows me to switch version and plug it into other losses no problem,
but for now, lets just use the MONAI losses.
"""

# example of how to use the MONAI losses:

from monai.losses import (
    DiceLoss, MaskedDiceLoss, GeneralizedDiceLoss, GeneralizedWassersteinDiceLoss, 
    FocalLoss, DiceFocalLoss, GeneralizedDiceFocalLoss, TverskyLoss, ContrastiveLoss
)

