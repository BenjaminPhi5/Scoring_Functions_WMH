"""

load a dataset

apply augmentation

setup the sampler

instantiate the dataloader

return dataloader

"""

from uloss_wmh.data_loading.sampler_2D import BinaryTargetClassBatchSampler
from torch.utils.data import DataLoader
from uloss_wmh.augmentation.aux_datasets import AugmentAndExtractDataset
from torch.utils.data import ConcatDataset

def WMH_Challenge_dataloader_pipeline(dataset, augmentation_pipeline, sampler, batch_size, num_iterations=None, target_class=None, target_prop=None, shuffle=True):
    """
    pipeline for a 2D dataloader for the WMH challenge dataset, loading the train and validation dataloaders, or the test dataloader

    dataset, a torch dataset
    
    augmentation_pipeline: series of transforms to be applied to the data, can be None.

    sampler: string: targeted | standard . targeted uses target_prop to ensure all batches contain target_prop proportion of the target class 
                standard just uses shuffling as usual
    
    num_iterations: how many batches should the batch sampler return per epoch. Used only when sampler = targeted.
    
    target_class: int 
    target_prop: none, if standard sampler used, or float in [0,1]

    shuffle: if sampler = standard, set to False for the dataset to not be shuffled.
    """

    assert target_prop == None or 0 <= target_prop <= 1

    dataset = AugmentAndExtractDataset(dataset, augmentation_pipeline, ['image', 'label'])

    if sampler == 'targeted':
        assert num_iterations != None
        dataloader = DataLoader(dataset, batch_sampler=BinaryTargetClassBatchSampler(dataset, target_class, target_prop, batch_size, num_iterations), num_workers=6, pin_memory=True)

    elif sampler == 'standard':
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=6, pin_memory=True)

    else:
        raise ValueError("sampler should be 'targeted' or 'standard'")

    return dataloader