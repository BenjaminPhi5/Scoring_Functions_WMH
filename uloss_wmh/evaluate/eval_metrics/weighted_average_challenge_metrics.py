"""
this computes the means of the performance data for each domain relative to its size in the dataset,
as done in the WMH challenge dataset.

I think for each mean metric I compute I probably need to do it in this way, it does make sense.
unless of course I separate the results per domain, which I may wish to do for the calibration statistics.
"""

import torch
    
def weighted_challenge_metrics(model_performances_per_domain):
    lengths = []
    mean_dices = []
    mean_f1s = []
    mean_avds = []
    mean_hd95s = []
    mean_recalls = []
    for data in model_performances_per_domain:
        lengths.append(len(data['dice'].tolist()))
        mean_dices.append(data['dice'].mean().item())
        mean_f1s.append(data['f1'].mean().item())
        mean_avds.append(data['avd'].mean().item())
        mean_hd95s.append(data['hd95'].mean().item())
        mean_recalls.append(data['recall'].mean().item())
        
    mean_dices = [m * lengths[i] for (i, m) in enumerate(mean_dices)]
    mean_f1s = [m * lengths[i] for (i, m) in enumerate(mean_f1s)]
    mean_avds = [m * lengths[i] for (i, m) in enumerate(mean_avds)]
    mean_hd95s = [m * lengths[i] for (i, m) in enumerate(mean_hd95s)]
    mean_recalls = [m * lengths[i] for (i, m) in enumerate(mean_recalls)]
    
    total = torch.Tensor(lengths).sum()
    
    return {
        "dice":torch.round(torch.Tensor(mean_dices).sum() / total, decimals=2).item(),
        "f1":torch.round(torch.Tensor(mean_f1s).sum() / total, decimals=2).item(),
        "avd":torch.round(torch.Tensor(mean_avds).sum() / total, decimals=2).item(),
        "hd95":torch.round(torch.Tensor(mean_hd95s).sum() / total, decimals=2).item(),
        "recall":torch.round(torch.Tensor(mean_recalls).sum() / total, decimals=2).item(),
    }
        