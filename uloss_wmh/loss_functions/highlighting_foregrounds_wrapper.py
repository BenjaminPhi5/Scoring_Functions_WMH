import torch.nn as nn

class HF_wrapper(nn.Module):
    def __init__(self, base_loss, N=6, weights=None):
        """
        N is the number of downsamples to be applied
        weights is the list of weightings for each component of the loss function.
        weights[0] is the weight applied to the full size output label, and weight.
        
        NOTE: if the label is not one-hot encoded then the class with the largest
        value will dominate. However, if the classes are one-hot this doesn't matter.
        However, this means it won't be a true one hot encoding anymore, where its possible
        for multiple classes to have a one at the same pixel (this is fine for Dice and brier based loss,
        cross_entropy won't like this, not sure what to do in this case...
        may need to force one class to dominate, which is fine for my data but not in general...
        Perhaps looking at the nn-Unet I will get an answer.
        """
        super().__init__()
        self.base_loss = base_loss
        self.N = N
        self.weights = torch.Tensor(weights)
        assert len(weights) == N
        
    def forward(self, outputs, label):
        labels = [label]
        
        dtype = label.dtype
        device = label.device
        label = label.type(torch.float32)
        
        for _ in range(self.N-1):
            label = nn.functional.max_pool2d(label, kernel_size=2)
            labels.append(label.type(dtype))
        labels = labels[::-1] # reverse so that smallest is at the front, as the models output smallest to largest.
        
        losses = [self.base_loss(f, l) for (f, l) in zip(outputs, labels)]
        
        losses = torch.stack(losses) * self.weights.to(device)
        
        return losses.sum()