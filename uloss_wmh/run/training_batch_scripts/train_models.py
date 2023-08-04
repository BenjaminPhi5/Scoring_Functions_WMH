import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch
from torchinfo import summary
import segmentation_models_pytorch as smp
from uloss_wmh.models.park_unet import HF_Unet

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
#from pytorch_lightning.callbacks import TQDMProgressBar
import os
import pytorch_lightning as pl

from ulw_data.torch_dataset.challenge_data_pipeline import  train_data_pipeline
from uloss_wmh.data_loading.WMH_Challenge_dataloading_pipeline import WMH_Challenge_dataloader_pipeline
from uloss_wmh.augmentation.nn_unet_augmentations import get_nnunet_transforms
from uloss_wmh.fitting.get_trainer import get_trainer
from uloss_wmh.fitting.fitter import StandardLitModelWrapper
from uloss_wmh.fitting.optimizer_constructor import OptimizerConfigurator, standard_configurations
from uloss_wmh.fitting.training_metrics import WrappedDiceMetric, WrappedDiceMetricFrom2Channel, WrappedDiceMetricNTo1Channel

# loss functions
from monai.losses import DiceLoss, GeneralizedDiceLoss, FocalLoss, TverskyLoss, DiceCELoss, DiceFocalLoss
from uloss_wmh.loss_functions.brier import Brier, BrierPlusDice
from uloss_wmh.loss_functions.odyssey_dice import SoftDiceLoss, DC_and_topk_loss
from uloss_wmh.loss_functions.odyssey_ND_Crossentropy import TopKLoss, CrossentropyND
from uloss_wmh.loss_functions.dice_plusplus import DicePlusPlusLoss

import torch.nn as nn
from  uloss_wmh.loss_functions.utils import normalize_inputs
import torch

import argparse

OPTIM_CONFIG = {
    0:OptimizerConfigurator(optim='Adam', lr=2e-4, weight_decay=0, scheduler='ReduceOnPlateau', patience=12, threshold=1e-4, factor=0.3, min_lr=1e-6, monitor='train_loss', verbose_lr=True),
    1:OptimizerConfigurator(optim='SGD', lr=2e-4, nesterov=True, momentum=0.9, weight_decay=0, scheduler='ReduceOnPlateau', patience=12, threshold=1e-4, factor=0.3, min_lr=1e-6, monitor='train_loss', verbose_lr=True),
    2:OptimizerConfigurator(optim='Adam', lr=3e-4, weight_decay=0, scheduler='ReduceOnPlateau', patience=12, threshold=1e-4, factor=0.3, min_lr=1e-6, monitor='val_loss', verbose_lr=True),
    3:OptimizerConfigurator(optim='SGD', lr=0.01, nesterov=True, momentum=0.9, weight_decay=0, scheduler='Polynomial', poly_power=0.9, total_epochs=1000, verbose_lr=True),
    4:OptimizerConfigurator(optim='Adam', lr=1e-8, weight_decay=0, scheduler='Warmup', monitor='train_loss', verbose_lr=True, total_epochs=1000, poly_power=0.9),
    5:OptimizerConfigurator(optim='Adam', lr=3e-4, weight_decay=0, scheduler='ReduceOnPlateau', patience=25, threshold=1e-4, factor=0.1, min_lr=1e-6, monitor='val_loss', verbose_lr=True),
    6:OptimizerConfigurator(optim='Adam', lr=3e-3, weight_decay=0, scheduler='ReduceOnPlateau', patience=12, threshold=1e-4, factor=0.3, min_lr=1e-6, monitor='train_loss', verbose_lr=True),
    7:OptimizerConfigurator(optim='Adam', lr=3e-5, weight_decay=0, scheduler='ReduceOnPlateau', patience=12, threshold=1e-4, factor=0.3, min_lr=1e-6, monitor='train_loss', verbose_lr=True),
    8:OptimizerConfigurator(optim='Adam', lr=3e-4, weight_decay=0, scheduler='ReduceOnPlateau', patience=1, threshold=1e-4, factor=0.1, min_lr=1e-6, monitor='val_loss', verbose_lr=True),
}


# class SphericalLossPerClass(nn.Module):
#     def __init__(self, include_background=False, sigmoid=False, softmax=True, alpha=2, weight=1, reduce=True):
#         super().__init__()
#         self.sigmoid = sigmoid
#         self.softmax = softmax
#         self.include_background = include_background
#         self.alpha = alpha
#         self.base_loss = SphericalLoss(sigmoid=False, softmax=False, alpha=self.alpha, weight=1, reduce=reduce)
#         self.weight = weight
        
#     def forward(self, predictions, targets):
#         predictions = normalize_inputs(predictions, self.sigmoid, self.softmax)
        
#         if not self.include_background and predictions.shape[1] != 1:
#             predictions = predictions[:,1:]
#             targets = targets[:,1:]
            
            
#         classes = predictions.shape[1]
        
#         loss = None
#         for c in range(classes):
#             p = predictions[:,c].unsqueeze(1)
#             t = targets[:,c].unsqueeze(1)
#             l = self.base_loss(p, t)
#             if loss == None:
#                 loss = l
#             else:
#                 loss += l
            
#         return self.weight * loss / classes
        

# class SphericalLoss(nn.Module):
#     def __init__(self, sigmoid=False, softmax=True, alpha=2, weight=1, reduce=True):
#         super().__init__()
#         self.sigmoid = sigmoid
#         self.softmax = softmax
#         self.alpha = alpha
#         self.weight = weight
#         self.reduce =reduce
        
#     def forward(self, predictions, targets):
#         predictions = normalize_inputs(predictions, self.sigmoid, self.softmax)
        
#         bs = predictions.shape[0]
#         classes = predictions.shape[1]
#         predictions = predictions.view(bs, classes, -1)
#         targets = targets.view(bs, classes, -1)
#         voxels = predictions.shape[-1]
        
#         # print(predictions.shape)
            
#         if classes == 1:
#             # print("run")
#             ohe_preds = torch.zeros((bs, 2, voxels), dtype=predictions.dtype, device=predictions.device)
#             ohe_targets = torch.zeros((bs, 2, voxels), dtype=targets.dtype, device=targets.device)
            
#             # print(ohe.shape)
#             ohe_preds[:,0] = 1 - predictions.squeeze()
#             ohe_preds[:,1] = predictions.squeeze()
            
#             ohe_targets[:,0] = 1 - targets.squeeze()
#             ohe_targets[:,1] = targets.squeeze()
            
            
#             predictions = ohe_preds
#             targets = ohe_targets
#             classes = 2
            
#         # print(predictions.shape, predictions.sum(dim=1).unique())
#         norm_term = predictions.pow(self.alpha).sum(dim=1).pow((self.alpha-1)/self.alpha)
#         # print("nt: ", norm_term.unique())
#         # print(norm_term[0,:5], norm_term.min(), norm_term.max())
#         target_class_predictions = torch.zeros((bs, voxels), device=predictions.device, dtype=predictions.dtype)
        
#         #print(target_class_predictions.shape, predictions.shape, targets.shape)
#         # print(target_class_predictions.sum())
#         for c in range(classes):
#             target_class_predictions[targets[:,c]==1] = predictions[:,c][targets[:,c]==1]
#             # print(target_class_predictions.sum())
            
#         # print("tc: ", target_class_predictions.unique())
            
#         spherical_score = target_class_predictions.pow(self.alpha-1) / norm_term
        
#         # print(spherical_score.unique())
#         # print((spherical_score==0).sum())
        
#         spherical_loss = 1. - spherical_score
        
#         if self.reduce:
#             return spherical_loss.sum(dim=1).mean() * self.weight
        
#         return spherical_loss#

class SphericalLossPerClass2(nn.Module):
    def __init__(self, include_background=False, sigmoid=False, softmax=True, weight=1, reduction="mean_sum"):
        super().__init__()
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.include_background = include_background
        self.base_loss = SphericalLoss2(sigmoid=False, softmax=False, weight=1, reduction=reduction)
        self.weight = weight
        
    def forward(self, predictions, targets):
        predictions = normalize_inputs(predictions, self.sigmoid, self.softmax)
        
        if not self.include_background and predictions.shape[1] != 1:
            predictions = predictions[:,1:]
            targets = targets[:,1:]
            
        bs = predictions.shape[0]
        classes = predictions.shape[1]
        predictions = predictions.view(bs, classes, -1)
        targets = targets.view(bs, classes, -1)
        voxels = predictions.shape[-1]
        
        loss = None
        for c in range(classes):
            p = torch.zeros((bs, 2, voxels), dtype=predictions.dtype, device=predictions.device)
            t = torch.zeros((bs, 2, voxels), dtype=targets.dtype, device=targets.device)
            p[:,1] = predictions[:,c]
            p[:,0] = 1 - predictions[:,c]
            t[:,1] = targets[:,c]
            t[:,0] = 1 - targets[:,c]
            l = self.base_loss(p, t)
            if loss == None:
                loss = l
            else:
                loss += l
            
        return self.weight * loss / classes
    
class SphericalLoss2(nn.Module):
    def __init__(self, sigmoid=False, softmax=True, weight=1, reduction="mean_sum"):
        super().__init__()
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, predictions, targets):
        predictions = normalize_inputs(predictions, self.sigmoid, self.softmax)
        bs = predictions.shape[0]
        
        spherical_loss = 1. - (predictions * targets).sum(dim=1) / predictions.norm(dim=1)
        
        if self.reduction == "mean":
            return spherical_loss.mean() * self.weight
        
        if self.reduction == "mean_sum":
            return spherical_loss.view(bs, -1).sum(dim=1).mean() * self.weight
        
        if self.reduction == "none" or self.reduction == None:
            return spherical_loss.view(bs, -1) * self.weight
    
class SphericalLossAlpha(nn.Module):
    def __init__(self, sigmoid=False, softmax=True, alpha=2, weight=1, reduction="mean_sum"):
        super().__init__()
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.alpha = alpha
        self.weight = weight
        self.reduction =reduction
        
    def forward(self, predictions, targets):
        predictions = normalize_inputs(predictions, self.sigmoid, self.softmax)
        bs = predictions.shape[0]
        
        alpha_normed = predictions.pow(self.alpha).sum(dim=1).pow((self.alpha-1)/self.alpha)
        spherical_loss = 1. - (predictions.pow(self.alpha-1) * targets).sum(dim=1) / alpha_normed
        
        if self.reduction == "mean":
            return spherical_loss.mean() * self.weight
        
        if self.reduction == "mean_sum":
            return spherical_loss.view(bs, -1).sum(dim=1).mean() * self.weight
        
        if self.reduction == "none" or self.reduction == None:
            return spherical_loss.view(bs, -1) * self.weight
    
class TopKWrapper(nn.Module):
    def __init__(self, base_loss, k=10, weight=1, sum_dim1=False):
        super().__init__()
        self.base_loss = base_loss
        self.k = k
        self.weight = weight
        self.sum_dim1 = sum_dim1
        
    def forward(self, predictions, targets):
        res = self.base_loss(predictions, targets)
        if self.sum_dim1:
            res = res.mean(dim=1)
        num_voxels = np.prod(res.shape)
        # print(res.shape)
        # print(num_voxels)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean() * self.weight
    
class FocalWrapper(nn.Module):
    def __init__(self, base_loss, gamma, weight=1, sigmoid=False, softmax=False, sum_dim1=False):
        super().__init__()
        self.base_loss = base_loss
        self.gamma = gamma
        self.weight = weight
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.sum_dim1 = sum_dim1
        
    def forward(self, predictions, targets):
        predictions = normalize_inputs(predictions, self.sigmoid, self.softmax)
        res = self.base_loss(predictions, targets)
        if self.sum_dim1:
            res = res.mean(dim=1)
            
        #print(res.mean())
            
        pt = (predictions * targets).sum(dim=1)
        pt = pt.view(pt.shape[0], -1)
        res = res.view(res.shape[0], -1)
        
        # print(pt.shape, res.shape)
        
        return ((1. - pt).pow(self.gamma) * res).mean() * self.weight
    
class ComboWrapper(nn.Module):
    def __init__(self, loss1, loss2):
        super().__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        
    def forward(self, predictions, targets):
        return self.loss1(predictions, targets) + self.loss2(predictions, targets)
    
class CrossEntropyPerClass(nn.Module):
    def __init__(self, weight, include_background=False):
        super().__init__()
        self.base_loss = CrossEntropyWrapper()
        self.weight = weight
        self.include_background=include_background
    
    def forward(self, predictions, targets):
        if not self.include_background and predictions.shape[1] != 1:
            predictions = predictions[:,1:]
            targets = targets[:,1:]
            
        bs = predictions.shape[0]
        classes = predictions.shape[1]
        predictions = predictions.view(bs, classes, -1)
        targets = targets.view(bs, classes, -1)
        voxels = predictions.shape[-1]
        
        loss = None
        for c in range(classes):
            p = torch.zeros((bs, 2, voxels), dtype=predictions.dtype, device=predictions.device)
            t = torch.zeros((bs, 2, voxels), dtype=targets.dtype, device=targets.device)
            p[:,1] = predictions[:,c]
            p[:,0] = 1 - predictions[:,c]
            t[:,1] = targets[:,c]
            t[:,0] = 1 - targets[:,c]
            l = self.base_loss(p, t)
            if loss == None:
                loss = l
            else:
                loss += l
            
        return self.weight * loss / classes
    
class CrossEntropyWrapper(nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self.weight = weight
        self.base_loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        return self.base_loss(predictions, targets.type(torch.float32)) * self.weight
    
    
def brier_score(pred):
    return 1 - (pred - 1).square()

def spherical_score(pred):
    return pred / (pred.square() + (1-pred).square()).sqrt()

def xent_score_2(pred):
    return 1 + pred.log2()

def xent_score_e(pred):
    return 1 + pred.log()

def xent_score_10(pred):
    return 1 + pred.log10()

def identity_score(pred):
    return pred
    
def scored_dice_from_types(pred, target, score_fn):
    bs = pred.shape[0]
    pred = pred.view(bs, -1)
    target = target.view(bs, -1)
    
    tps = ((target==1) * score_fn(pred)).sum(dim=1)
    fps = ((target==0) * score_fn(pred)).sum(dim=1)
    fns = ((target==1) * score_fn(1-pred)).sum(dim=1)
    
    return (1 - (2*tps + 1) / (2 * tps + fps + fns + 1)).mean()
    
class ScoredDice(nn.Module):
    def __init__(self, sigmoid=False, softmax=False, include_background=False, score="logarithmic"):
        super().__init__()
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.include_background = include_background
        
        if score=="logarithmic":
            self.score=xent_score_e
        elif score=="brier":
            self.score=brier_score
        elif score=="spherical":
            self.score=spherical_score
        else:
            raise ValueError
        
    def forward(self, predictions, targets):
        predictions = normalize_inputs(predictions, self.sigmoid, self.softmax)
        if not self.include_background and predictions.shape[1] != 1:
            predictions = predictions[:,1:]
            targets = targets[:,1:]
            
        classes = predictions.shape[1]
        
        loss = None
        for c in range(classes):
            p = predictions[:,c].unsqueeze(1)
            t = targets[:,c].unsqueeze(1)
            l = scored_dice_from_types(p, t, self.score)
            #print(l)
            if loss == None:
                loss = l
            else:
                loss += l
            
        return loss / classes
    
class CrossEntropyCustomLog(nn.Module):
    def __init__(self, log, reduction='mean', weight=1., apply_softmax=True):
        super().__init__()
        log = str(log)
        if log == "10":
            self.log_func = torch.log10
        elif log == "2":
            self.log_func = torch.log2
        
        elif log == "e":
            self.log_func = torch.log
        self.nllloss = nn.NLLLoss(reduction=reduction)
        self.apply_softmax=True
        self.weight=weight
            
    def forward(self, predictions, targets):
        targets = targets.type(torch.float32)
        if self.apply_softmax:
            predictions = torch.softmax(predictions, dim=1)
        
        predictions = self.log_func(predictions)
        targets = targets.argmax(dim=1)
        
        #print(predictions.shape, targets.shape)
        
        return self.nllloss(predictions, targets) * self.weight
    
class BrierPower(nn.Module):
    def __init__(self, include_background=False, sigmoid=False, softmax=False, weight=1, reduction="mean_sum", power=2.):
        super().__init__()
        self.include_background=include_background
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.weight = weight
        self.reduction = reduction
        self.power=power

    def forward(self, predictions, targets):
        predictions = normalize_inputs(predictions, self.sigmoid, self.softmax)
        bs = predictions.shape[0]
        classes = predictions.shape[1]
        predictions = predictions.view(bs, classes, -1)
        targets = targets.view(bs, classes, -1)

        if not self.include_background and predictions.shape[1] != 1:
            predictions = predictions[:,1:]
            targets = targets[:,1:]
            
        brier_power_loss = (predictions - targets).abs().pow(self.power)
        
        if self.reduction == "mean_sum":
            return brier_power_loss.sum(dim=-1).mean() * self.weight
        if self.reduction == "mean":
            return brier_power_loss.mean() * self.weight
        
        return brier_power_loss

    
PROPORTIONAL_REWEIGHTING_FACTOR = 357
LOSS_FUNCTION_CONFIG = {
    # training roadmap experiment 1 (questions 1, 2, 4b and 8)
    "dice":DiceLoss(softmax=True, include_background=False), 
    "dice_plusplus_gamma2":DicePlusPlusLoss(softmax=True, gamma=2, include_background=False),
    "dice_plusplus_gamma3":DicePlusPlusLoss(softmax=True, gamma=3, include_background=False),
    "dice_plusplus_gamma4":DicePlusPlusLoss(softmax=True, gamma=4, include_background=False),
    "spherical":SphericalLoss2(softmax=True, reduction='mean', weight=PROPORTIONAL_REWEIGHTING_FACTOR),
    "spherical_perclass":SphericalLossPerClass2(softmax=True, reduction='mean', include_background=False, weight=PROPORTIONAL_REWEIGHTING_FACTOR),
    "spherical_perclass_with_background":SphericalLossPerClass2(softmax=True, reduction='mean', include_background=True, weight=PROPORTIONAL_REWEIGHTING_FACTOR),
    "xent":CrossEntropyWrapper(),
    "xent_perclass":CrossEntropyPerClass(weight=PROPORTIONAL_REWEIGHTING_FACTOR/2),
    "xent_perclass_with_background":CrossEntropyPerClass(weight=PROPORTIONAL_REWEIGHTING_FACTOR/2, include_background=True),
    "brier":Brier(softmax=True),
    "brier_with_background":Brier(softmax=True, include_background=True),
    "scored_dice_logarithmic":ScoredDice(softmax=True, score='logarithmic'),
    "scored_dice_quadratic":ScoredDice(softmax=True, score='brier'),
    "scored_dice_spherical":ScoredDice(softmax=True, score='spherical'),
    
    # training roadmap experiment 2 (question 3)
    "xent_10":CrossEntropyCustomLog("10", weight=PROPORTIONAL_REWEIGHTING_FACTOR/2),
    "xent_e":CrossEntropyCustomLog("e", weight=PROPORTIONAL_REWEIGHTING_FACTOR/2),
    "xent_2":CrossEntropyCustomLog("2", weight=PROPORTIONAL_REWEIGHTING_FACTOR/2),
    "spherical_alpha1.5":SphericalLossAlpha(softmax=True, alpha=1.5, reduction="mean", weight=PROPORTIONAL_REWEIGHTING_FACTOR),
    "spherical_alpha2":SphericalLossAlpha(softmax=True, alpha=2, reduction="mean", weight=PROPORTIONAL_REWEIGHTING_FACTOR),
    "spherical_alpha2.5":SphericalLossAlpha(softmax=True, alpha=2.5, reduction="mean", weight=PROPORTIONAL_REWEIGHTING_FACTOR),
    "spherical_alpha3":SphericalLossAlpha(softmax=True, alpha=3, reduction="mean", weight=PROPORTIONAL_REWEIGHTING_FACTOR),
    "brier_power1.5":BrierPower(include_background=True, softmax=True, reduction="mean", power=1.5, weight=PROPORTIONAL_REWEIGHTING_FACTOR),
    "brier_power2":BrierPower(include_background=True, softmax=True, reduction="mean", power=2, weight=PROPORTIONAL_REWEIGHTING_FACTOR),
    "brier_power2.5":BrierPower(include_background=True, softmax=True, reduction="mean", power=2.5, weight=PROPORTIONAL_REWEIGHTING_FACTOR),
    "brier_power3":BrierPower(include_background=True, softmax=True, reduction="mean", power=3, weight=PROPORTIONAL_REWEIGHTING_FACTOR),
    
    # training roadmap experiment 3 (question 4a, 5 6,and 12)
    "spherical_topk10":TopKWrapper(base_loss=SphericalLossAlpha(softmax=True, alpha=2, reduction="none", weight=1), k=10, weight=PROPORTIONAL_REWEIGHTING_FACTOR),
    "spherical_topk50":TopKWrapper(base_loss=SphericalLossAlpha(softmax=True, alpha=2, reduction="none", weight=1), k=50, weight=PROPORTIONAL_REWEIGHTING_FACTOR),
    "spherical_topk90":TopKWrapper(base_loss=SphericalLossAlpha(softmax=True, alpha=2, reduction="none", weight=1), k=90, weight=PROPORTIONAL_REWEIGHTING_FACTOR),
    "brier_topk10":TopKWrapper(BrierPower(include_background=True, softmax=True, reduction="none", power=2, weight=1), k=10, sum_dim1=True, weight=PROPORTIONAL_REWEIGHTING_FACTOR),
    "brier_topk50":TopKWrapper(BrierPower(include_background=True, softmax=True, reduction="none", power=2, weight=1), k=50, sum_dim1=True, weight=PROPORTIONAL_REWEIGHTING_FACTOR),
    "brier_topk90":TopKWrapper(BrierPower(include_background=True, softmax=True, reduction="none", power=2, weight=1), k=90, sum_dim1=True, weight=PROPORTIONAL_REWEIGHTING_FACTOR),
    "xent_topk10":TopKWrapper(CrossEntropyCustomLog(log="e", reduction="none"), k=10, weight=PROPORTIONAL_REWEIGHTING_FACTOR/2),
    "xent_topk50":TopKWrapper(CrossEntropyCustomLog(log="e", reduction="none"), k=50, weight=PROPORTIONAL_REWEIGHTING_FACTOR/2),
    "xent_topk90":TopKWrapper(CrossEntropyCustomLog(log="e", reduction="none"), k=90, weight=PROPORTIONAL_REWEIGHTING_FACTOR/2),
    "spherical_focal0.5":FocalWrapper(SphericalLossAlpha(softmax=False, sigmoid=False, weight=1, alpha=2, reduction='none'), gamma=0.5, weight=PROPORTIONAL_REWEIGHTING_FACTOR, softmax=True),
    "spherical_focal1.0":FocalWrapper(SphericalLossAlpha(softmax=False, sigmoid=False, weight=1, alpha=2, reduction='none'), gamma=1.0, weight=PROPORTIONAL_REWEIGHTING_FACTOR, softmax=True),
    "spherical_focal1.5":FocalWrapper(SphericalLossAlpha(softmax=False, sigmoid=False, weight=1, alpha=2, reduction='none'), gamma=1.5, weight=PROPORTIONAL_REWEIGHTING_FACTOR, softmax=True),
    "brier_focal0.5":FocalWrapper(BrierPower(softmax=False, sigmoid=False, weight=1, power=2, reduction='none', include_background=True), gamma=0.5, sum_dim1=True, weight=PROPORTIONAL_REWEIGHTING_FACTOR, softmax=True),
    "brier_focal1.0":FocalWrapper(BrierPower(softmax=False, sigmoid=False, weight=1, power=2, reduction='none', include_background=True), gamma=1.0, sum_dim1=True, weight=PROPORTIONAL_REWEIGHTING_FACTOR, softmax=True),
    "brier_focal1.5":FocalWrapper(BrierPower(softmax=False, sigmoid=False, weight=1, power=2, reduction='none', include_background=True), gamma=1.5, sum_dim1=True, weight=PROPORTIONAL_REWEIGHTING_FACTOR, softmax=True),
    "xent_focal0.5":FocalWrapper(CrossEntropyCustomLog(log="e", reduction="none", apply_softmax=False), gamma=0.5, weight=PROPORTIONAL_REWEIGHTING_FACTOR/2, softmax=True),
    "xent_focal1.0":FocalWrapper(CrossEntropyCustomLog(log="e", reduction="none", apply_softmax=False), gamma=1.0, weight=PROPORTIONAL_REWEIGHTING_FACTOR/2, softmax=True),
    "xent_focal1.5":FocalWrapper(CrossEntropyCustomLog(log="e", reduction="none", apply_softmax=False), gamma=1.5, weight=PROPORTIONAL_REWEIGHTING_FACTOR/2, softmax=True),
    "xent_unscaled":CrossEntropyCustomLog("e", weight=1),
    
    # training roadmap experiment 4 (questions 7 and 9)
    "spherical_topk1":TopKWrapper(base_loss=SphericalLossAlpha(softmax=True, alpha=2, reduction="none", weight=1), k=1, weight=PROPORTIONAL_REWEIGHTING_FACTOR/100),
    "brier_topk1":TopKWrapper(BrierPower(include_background=True, softmax=True, reduction="none", power=2, weight=1), k=1, sum_dim1=True, weight=PROPORTIONAL_REWEIGHTING_FACTOR/100),
    "xent_topk1":TopKWrapper(CrossEntropyCustomLog(log="e", reduction="none"), k=1, weight=PROPORTIONAL_REWEIGHTING_FACTOR/200),
    "spherical_topk0.5":TopKWrapper(base_loss=SphericalLossAlpha(softmax=True, alpha=2, reduction="none", weight=1), k=0.5, weight=PROPORTIONAL_REWEIGHTING_FACTOR/150),
    "brier_topk0.5":TopKWrapper(BrierPower(include_background=True, softmax=True, reduction="none", power=2, weight=1), k=0.5, sum_dim1=True, weight=PROPORTIONAL_REWEIGHTING_FACTOR/150),
    "xent_topk0.5":TopKWrapper(CrossEntropyCustomLog(log="e", reduction="none"), k=0.5, weight=PROPORTIONAL_REWEIGHTING_FACTOR/300),
    
    # old experiment losses
    "topk":TopKLoss(k=10), 
    "focal":FocalLoss(gamma=2.0, reduction='sum'), 
    "dice_focal_2":DiceFocalLoss(softmax=True, gamma=2.0, include_background=False),
    "dice_focal_3":DiceFocalLoss(softmax=True, gamma=3.0, include_background=False),
    "dice_focal_4":DiceFocalLoss(softmax=True, gamma=4.0, include_background=False),
    "tversky_alpha.7":TverskyLoss(softmax=True, alpha=0.7, beta=0.5, include_background=False),  #TODO DOES THE TVERSKY LOSS IGNORE THE BACKGROUND
    "tversly_beta.7":TverskyLoss(softmax=True, alpha=0.5, beta=0.7, include_background=False), 
    "soft_dice":SoftDiceLoss(apply_nonlin=lambda x : torch.softmax(x, 1), do_bg=False, smooth=1e-5), 
    "dice_brier":BrierPlusDice(softmax=True, brier_factor=1/100, dice_factor=1, include_background=False), 
    "dice_cross_entropy":DiceCELoss(softmax=True, include_background=True), 
#     "spherical":SphericalLoss(softmax=True, weight=1, reduce=True),
#     "spherical_per_class":SphericalLossPerClass(softmax=True, weight=0.01, reduce=True),
#     "spherical_topk_per_class":TopKWrapper(SphericalLossPerClass(softmax=True, weight=0.01, reduce=False), weight=10),
#     "spherical_topk":TopKWrapper(SphericalLoss(softmax=True, weight=1, reduce=False), weight=10),
#     "spherical_topk_dice":ComboWrapper(DiceLoss(softmax=True, include_background=False), TopKWrapper(SphericalLossPerClass(softmax=True, weight=1, reduce=False), weight=10)),
#     "spherical_topk_diceplusplus":ComboWrapper(DicePlusPlusLoss(softmax=True, gamma=2), TopKWrapper(SphericalLossPerClass(softmax=True, weight=1, reduce=False), weight=10)),
#     "brier_topk_dice_plusplus":ComboWrapper(
#     DicePlusPlusLoss(softmax=True, include_background=False, gamma=2),
#     TopKWrapper(Brier(include_background=False, softmax=True, reduce=False, weight=2))
# ),
#     "sperhical_dice":ComboWrapper(SphericalLossPerClass(softmax=True, weight=0.01, reduce=True), DiceLoss(softmax=True, include_background=False)),
#     "spherical_diceplusplus": ComboWrapper(SphericalLossPerClass(softmax=True, weight=0.01, reduce=True), DicePlusPlusLoss(softmax=True, include_background=False, gamma=2)),
    "dice_topk":DC_and_topk_loss({"do_bg":False}, {"k":10}),
    
}


def construct_parser():
    parser = argparse.ArgumentParser(description="initial global loss function tests")
    
    parser.add_argument('--ds_dir', default="/media/benp/NVMEspare/datasets/full_WMH_Chal_dataset_norm05/preprocessed/collated/", type=str)
    parser.add_argument('--ckpt_dir', default='', type=str)
    parser.add_argument('--val_prop', default=0.15, type=float)
    parser.add_argument('--seed', default=5, type=int)
    parser.add_argument('--max_epochs', default=1000, type=int)
    parser.add_argument('--model_type', default='standard', type=str)
    parser.add_argument('--do_cv', default="false", type=str)
    parser.add_argument('--cv_fold', default=0, type=int)
    parser.add_argument('--cv_splits', default='5', type=str)
    parser.add_argument('--use_both_classes', default="true", type=str)
    parser.add_argument('--remove_mask_channel', default="true", type=str)
    parser.add_argument('--optim_schedule', default=5, type=int)
    parser.add_argument('--use_full_precision', default="false", type=str)
    parser.add_argument('--early_stop_patience', default=50, type=int)
    parser.add_argument('--loss_name', default='dice_brier', type=str)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--id', default='', type=str)
    
    return parser


def main(args):
    
    collated_folder = args.ds_dir
    ckpt_dir = args.ckpt_dir
    val_prop = args.val_prop
    seed = args.seed
    max_epochs = args.max_epochs
    
    model_type = args.model_type
    do_cv = args.do_cv.lower() == "true"
    cv_fold = args.cv_fold
    cv_splits = args.cv_splits
    two_class = args.use_both_classes.lower() == "true"
    remove_mask_channel = args.remove_mask_channel.lower() == "true"
    optim_schedule_ID = args.optim_schedule
    folder_name = f"{args.loss_name}_{args.id}"#_{args.optim_schedule}"
    
    precision = 32 if args.use_full_precision.lower() == "true" else "16-mixed"
    early_stop_patience=args.early_stop_patience
    
    if not do_cv:
        cv_splits = None
        cv_fold = None
    
    
    print(
        f"remove mask channel {remove_mask_channel}\n"
        + f"train on both classes {two_class}\n"
        + f"model type {model_type}\n"
        + f"do_cv {do_cv}\n"
        + f"cv fold and splits {cv_fold} {cv_splits}\n"
        + f"optim schedule id: {optim_schedule_ID}\n"
        + f"model output folder name {folder_name}\n"
        + f"early stop patience {early_stop_patience}\n"
        + f"precision {precision}\n"
    )
    
    model_folder = ckpt_dir + folder_name + "/"
    
    print("loading training data")
    train_ds, val_ds = train_data_pipeline(collated_folder, val_proportion=val_prop, seed=seed, transforms=None, dims=2, remove_mask_channel=remove_mask_channel, cv_fold=cv_fold, cv_splits=cv_splits)
    print("shape: ", train_ds[0]['image'].shape)
    
    print(len(train_ds), len(val_ds))
    
    if two_class:
        one_hot_encode = True
        outchannels = 3
    else:
        one_hot_encode = False
        outchannels = 1
    
    if remove_mask_channel:
        inchannels = 2
    else:
        inchannels = 3
    
    augmentations = get_nnunet_transforms(axial_only=False, dims=2, out_spatial_dims=(192,224), one_hot_encode=one_hot_encode, num_classes=3, target_class=1)
    
    print("building data loaders")
    train_dl = WMH_Challenge_dataloader_pipeline(
        train_ds,
        augmentation_pipeline=augmentations,
        sampler='standard',
        batch_size=32,
        num_iterations=125,
        # target_class=1,
        # target_prop=0.33,
    )
    
    val_dl = WMH_Challenge_dataloader_pipeline(
        val_ds,
        augmentation_pipeline=augmentations,
        sampler='standard',
        batch_size=32,
        shuffle=False
    )
        
    model_base = smp.Unet(
        encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=inchannels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=outchannels,                      # model output channels (number of classes in your dataset)
    )
    
    loss_func = LOSS_FUNCTION_CONFIG[args.loss_name]
    
    custom_optim_config = OPTIM_CONFIG[optim_schedule_ID]
    
    model = StandardLitModelWrapper(model_base, loss_func, custom_optim_config, logging_metric=None)
    
    
    accelerator="gpu"
    devices=1
    use_early_stopping=True
    early_stop_on_train=False


    checkpoint_callback = ModelCheckpoint(model_folder, save_top_k=2, monitor="val_loss")

    callbacks = [checkpoint_callback]
    if use_early_stopping:

        if early_stop_on_train:
            early_stop_callback = EarlyStoppingOnTraining(monitor="train_loss", min_delta=1e-4, patience=early_stop_patience, verbose="False", mode="min", check_finite=True)
        else:
            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=early_stop_patience, verbose="False", mode="min", check_finite=True)
        callbacks.append(early_stop_callback)

    trainer = pl.Trainer(
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        precision=precision,
        default_root_dir=model_folder,
    )
    
    trainer.fit(model, train_dl, val_dl)
    trainer.validate(model, val_dl, ckpt_path='best')

if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)