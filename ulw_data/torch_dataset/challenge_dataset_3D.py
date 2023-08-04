import numpy as np
import os
from torch.utils.data import Dataset, ConcatDataset
from torch import from_numpy
from torch import float32, long, LongTensor
from ulw_data.torch_dataset.aux_datasets import RemoveMaskDs, PutMaskInLabelDs3D

class WMH_Challenge_Dataset_3D(Dataset):
    def __init__(self, data_file):
        super().__init__()
        data = np.load(data_file)
        self.uids = data['kwds']
        img_data = from_numpy(data['args'])
        self.imgs = img_data[:,0:3].type(float32)
        self.labels = img_data[:,3].type(long)
        
    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        uid = self.uids[idx]

        # returns (flair, t1, mask), (wmh), uid
        return {
            "image":img,
            "label":label.unsqueeze(dim=0),
            "uid":uid
        }

    def __len__(self):
        return self.imgs.shape[0]

def load_3D_wmh_chal(ds_path, train=True, combine=False, remove_mask_channel=False, put_mask_in_label=False):
    domain_files = {
        "train": [
            'Amsterdam_GE3T_training.npz',
            'Singapore_training.npz',
            'Utrecht_training.npz',
        ],
        "test": [
            'Amsterdam_GE1T5_test.npz',
            'Amsterdam_GE3T_test.npz',
            'Amsterdam_Philips_VU_PETMR_01_test.npz',
            'Singapore_test.npz',
            'Utrecht_test.npz',
        ]
    }

    if train:
        domains = domain_files['train']
    else:
        domains = domain_files['test']
        
    datasets = [
       WMH_Challenge_Dataset_3D(f"{ds_path}{domain}") for domain in domains
    ]
    
    if put_mask_in_label:
        datasets = [PutMaskInLabelDs3D(ds, remove_mask_channel) for ds in datasets]
    elif remove_mask_channel:
        datasets = [RemoveMaskDs(ds) for ds in datasets]

    if combine:
        return ConcatDataset(datasets)

    return datasets
    
















