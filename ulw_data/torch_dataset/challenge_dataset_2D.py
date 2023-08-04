"""
What do I want from a 2D dataset? That is a good question. I think for now we can just make a 2D dataset
of slices,

I shall just keep the slices that contain brain. and I can cull stochastically some slices that
do not have WMH perhaps? Or I could keep it simple for now, lets just keep it to slices that contain brain

how am I going to split into train,val,test.

Well I want to have the validation use separate individuals. 
so I need to split into train and val first,
and I would like an equal number of validation from each domain.
So I need to:

- load per domain in 3D
- split train into train and validate
- combine together each domain per split
- convert to 2D
- apply augmentation pipeline
- create a dataloader
"""

import numpy as np
import os
from torch.utils.data import Dataset
from torch import where

class WMH_Challenge_Dataset_2D(Dataset):
    def __init__(self, base_3d_dataset, transforms, remove_mask_channel=False):
        self.transforms = transforms
        self.base_3d_dataset = base_3d_dataset
        self.transforms = transforms
        self.idx_slice_pairs = []
        self.remove_mask_channel = remove_mask_channel

        # calculate the slices that I should consider
        # for now this dataset just gives us axial slices but I could
        # consider other slices (which would give me a larger dataset but anyway...)
        for idx, data in enumerate(self.base_3d_dataset):
            imgs = data['image']
            mask = imgs[2]
            brain_locations = where(mask)
            min_axial_slice = brain_locations[2].min()
            max_axial_slice = brain_locations[2].max()
            slice_range = range(min_axial_slice, max_axial_slice+1)
            self.idx_slice_pairs.extend(
                list(
                    zip(
                        [idx for _ in slice_range],
                        slice_range
                    )
                )
            )
                                

    def __getitem__(self, idx): 
        base_idx, slice_idx = self.idx_slice_pairs[idx]
        data = self.base_3d_dataset[base_idx]
        imgs = data['image']
        if self.remove_mask_channel:
            imgs = imgs[0:2] # channel is the first dimension for 3D image obeject
        wmh = data['label']
        uid = data['uid']

        
        imgs_2d = imgs[:,:,:,slice_idx]
        wmh_2d = wmh[:,:,:,slice_idx]
        uid_slice = f"{uid}_{slice_idx}"
        
        if self.transforms:
            imgs_2d, wmh_2d = self.transforms(imgs_2d, wmh_2d)
            
        
        return {
            "image":imgs_2d,
            "label":wmh_2d,
            "uid":uid_slice,
        }

    def __len__(self):
        return len(self.idx_slice_pairs)