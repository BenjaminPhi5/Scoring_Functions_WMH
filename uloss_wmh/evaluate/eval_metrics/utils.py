import torch
from tqdm import tqdm
from kornia.morphology import dilation

def get_preds(model, test_datasets, sigmoid=False, softmax=False):
    outputs = []
    
    with torch.no_grad():
        for dataset in test_datasets:
            for data in tqdm(dataset):
                inp = data['image'].cuda().moveaxis(-1, 0)
                out = model.cuda()(inp)
                
                if sigmoid:
                    out = torch.sigmoid(out).squeeze()
                elif softmax:
                    out = torch.softmax(out, dim=1)[:, 1].squeeze()
                else:
                    out = out.squeeze()
                
                out = out.moveaxis(0, -1).cpu()
                outputs.append(out)
                
    return outputs

def get_inputs(test_datasets):
    inputs = []
    for dataset in test_datasets:
        for data in dataset:
            inputs.append(data['image'])
    return inputs

def get_labels(test_datasets):
    labels = []
    for dataset in test_datasets:
        for data in dataset:
            labels.append(data['label'].squeeze())
    return labels

def get_brain_voxels(input, pred):
    return pred[input[2] == 1]

def dilate_considered_area(pred_binary, label, threshold, steps=0):
    step0_area = (pred_binary > threshold) | (label == 1)
    step0_area = step0_area.moveaxis(-1,0).unsqueeze(1) # step0_area requries shape B C H W
    
    result = step0_area.type(torch.float32)
    for step in range(steps):
        result = dilation(result, kernel=torch.ones(3, 3), engine='convolution')
    
    return result.squeeze().moveaxis(0, -1).type(torch.long)

def get_n_step_dilated_voxels(pred, label, threshold, steps):
    dilated = dilate_considered_area(pred, label, threshold=threshold, steps=steps)
    return pred[dilated==1]

