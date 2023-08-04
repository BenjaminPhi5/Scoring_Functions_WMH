# precision and recall curve as the confidence threshold is varied
def precision_recall_curves(predictions, labels, num_bins=20):
    # TODO: I need to make this a weighted precision and recall at somepoint, and include does
    # with the metrics that I vary
    bins = np.linspace(0, 1, num_bins)
    
    precisions = []
    recalls = []
    
    for bin in tqdm(bins):
        tps = 0
        fps = 0
        fns = 0
        for i in range(len(predictions)):
            pred = predictions[i].cuda()
            label = labels[i].cuda()
            tps += ((pred > bin) * (label==1)).sum().item()
            fps += ((pred > bin) * (label==0)).sum().item()
            fns += ((pred <= bin) * (label==1)).sum().item()
            
        precisions.append(tps / (tps + fps + 1e-8))
        recalls.append(tps / (tps + fns + 1e-8))
            
        
    return bins, precisions, recalls