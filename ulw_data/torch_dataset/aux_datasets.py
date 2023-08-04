from torch.utils.data import Dataset
import torch

class RemoveMaskDs(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        data['image'] = data['image'][0:2] # removes the mask channel!
        return data
    
class PutMaskInLabelDs3D(Dataset):
    def __init__(self, base_dataset, remove_mask=False):
        # base dataset should be a 3D dataset.
        # this dataset moves the mask channel to the label as the last channel
        self.base_dataset = base_dataset
        self.remove_mask = remove_mask
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        image = data['image']
        mask = image[2].unsqueeze(0)
        # print(mask.shape, data['label'].shape)
        if self.remove_mask:
            data['image'] = data['image'][0:2] # removes the mask channel!
        data['label'] = torch.cat([data['label'], mask], dim=0) # puts the mask channel as the last channel in the label dataset 
        return data