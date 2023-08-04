# now compute the UEO, sUEO and sUEO score....
def sUEO(pred, ent_map, target):
    errors = (pred != target)
    
    numerator = 2 * (ent_map * errors).sum()
    denominator = (errors**2).sum() + (ent_map**2).sum()
    
    return (numerator / denominator).item()

# def sUEO_new_version(pred, ent_map, target)