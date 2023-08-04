import torch

def fast_dice(pred, target):
    p1 = (pred == 1)
    t1 = (target == 1)
    intersection = (pred == 1) & (target == 1)
    numerator = 2 * intersection.sum()
    denominator = p1.sum() + t1.sum()
    return (numerator/(denominator + 1e-30)).item()


def filtered_dice(pred, ent_map, target, threshold):
    uncertain_locs = ent_map < threshold
    remaining_pred = pred[uncertain_locs]
    remaining_target = target[uncertain_locs]
    
    return fast_dice(remaining_pred, remaining_target)

def filtered_tps_score(pred, ent_map, target, threshold):
    total_tps = ((pred == 1) & (target == 1)).sum().item()
    uncertain_locs = ent_map < threshold
    
    filtered_tps = ((pred[uncertain_locs] == 1) & (target[uncertain_locs] == 1)).sum().item()
    
    return (total_tps - filtered_tps) / (total_tps + 1e-30)

def filtered_tns_score(mask, pred, ent_map, target, threshold):
    mask = mask.type(torch.bool)
    pred = pred[mask]
    ent_map = ent_map[mask]
    target = target[mask]
    
    total_tns = ((pred == 0) & (target == 0)).sum().item()
    uncertain_locs = ent_map < threshold
    
    filtered_tns = ((pred[uncertain_locs] == 0) & (target[uncertain_locs] == 0)).sum().item()
    
    return (total_tns - filtered_tns) / (total_tns + 1e-30)


def bras():
    # TODO
    pass