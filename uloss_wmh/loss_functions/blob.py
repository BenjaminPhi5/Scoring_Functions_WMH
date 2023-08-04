import torch.nn as nn
import torch
from uloss_wmh.loss_functions.utils import normalize_inputs
import warnings

class BlobLoss:
    pass

class BinaryBatchBlobLoss:
    pass

class BatchBlobLoss(nn.Module):
    def __init__(self, base_loss, sigmoid=False, softmax=True):
        print("TODOs for BlobLoss:\n1) Implement a version that computes the loss for each element of the batch separately\n2) Modify or add another loss term " +
              "that inlcudes the background class (currently loss is just 0 if there are no instances in the ground truth regardless of what the model predicts (" + 
              "although this will be included in any global loss function term\n3) Implememt a version that allows weighting based on lesion size")
              
        """
        base_loss is a binary classification loss that is applied to each instance for each class.
        """
        super().__init__()
        self.sigmoid=sigmoid
        self.softmax=softmax
        self.base_loss = base_loss

    def compute_masked_predictions_and_label(self, predictions, targets, instance_id):
        """
        the mask simply blocks out the pixels of every label that
        is not the target label.

        returns the mask that masks out all other labels,
        the target label
        """
        mask = torch.logical_not((targets != 0) & (targets != instance_id))

        return mask * predictions, mask * targets

    def forward(self, predictions, targets):
        """
        targets here is of shape [B,C,<dims>], where each instance gives a unique id
        to all the pixels in that instance. Each unique ID should be greater than 0.
        WARNING: each element in the batch needs to have a unique ID!
        """
        predictions = normalize_inputs(predictions, self.sigmoid, self.softmax)

        channels = targets.shape[1]
        if channels == 1:
            raise ValueError("Implicit background class not permitted. For binary blob loss, use the BatchBinaryBlobLoss or BinaryBlobLoss class")
            
        losses = [[] for _ in range(1, channels)]

        # we need to iterate over each element in the batch separately....
        # it should be sum over N sum over C I think, or does it if I treat each instance as separate?
        # this is an interesting point, I can just calculate it over the whole batch.....
        for b in range(predictions.shape[0]):
            for channel in range(1, channels): # skip the background channel.
                preds_channel = predictions[b, channel].unsqueeze(0)
                targets_channel = targets[b, channel].unsqueeze(0)
                ids = targets_channel.unique()
                class_ids = []
                losses = []
        
                for instance_id in ids:
                    if instance_id == 0: # 0 is the background class.
                        continue
        
                    masked_preds, instance_label = self.compute_mask_and_label(preds_channel, targets_channel, instance_id)
        
                    losses[channel].append(self.base_loss(masked_preds, instance_label))

        # not all batches might have an example of a particular class, so no instance losses are incurred here
        losses = [torch.Tensor(class_losses) for class_losses in losses]
        
        loss = 0
        for class_losses in losses:
            if len(losses) > 0:
                loss += class_losses.mean()

        loss /= len(losses)
        
        return loss
        
            
class BatchBinaryBlobLoss(BlobLoss):
    def __init__(self, base_loss, sigmoid=True):
        super().__init__(base_loss, sigmoid=sigmoid, softmax=False)

    def forward(self, predictions, targets):
        predictions = normalize_inputs(predictions, self.sigmoid, self.softmax)

        channels = targets.shape[1]
        if channels != 1:
            raise ValueError("Only binary task permitted. For MultiClass blobloss use the BatchBlobLoss or BlobLoss class")

        losses = []


        for b in range(predictions.shape[0]): # iterate over each element of the batch separately since ids might be shared between instances
            preds_b = predictions[b].unsqueeze(dim=0)
            targets_b = targets[b].unsqueeze(dim=0)
            
            ids = targets_b.unique()
        
            for instance_id in ids:
                if instance_id == 0:
                    continue

                masked_preds, instance_label = self.compute_mask_and_label(preds_channel, targets_channel, instance_id)
        
                losses.append(self.base_loss(masked_preds, instance_label))

        if len(losses) > 0:
            loss = torch.mean(loss)
        else:
            warnings.warn("batch had no ground truth... instances found within it, loss = 0 for this batch")
            loss = 0
            


def calculate_blob_loss_components(labels):
    pass

def get_blob_loss_Dataset(dataset):
    """
    this will take a torch dataset TODO formatting
    and return the labels as connected components (separated per class...)
    """
    pass