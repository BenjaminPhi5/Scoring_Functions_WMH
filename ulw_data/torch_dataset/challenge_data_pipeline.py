from ulw_data.torch_dataset.challenge_dataset_2D import WMH_Challenge_Dataset_2D
from ulw_data.torch_dataset.challenge_dataset_3D import WMH_Challenge_Dataset_3D, load_3D_wmh_chal
from ulw_data.torch_dataset.splits import train_val_splits, k_fold_challenge_splitter
from ulw_data.torch_dataset.aux_datasets import RemoveMaskDs, PutMaskInLabelDs3D
from torch.utils.data import ConcatDataset

def train_data_pipeline(ds_path, val_proportion, seed, transforms, dims=2, remove_mask_channel=False, put_mask_in_label=False, cv_fold=None, cv_splits=None):
    datasets = load_3D_wmh_chal(ds_path, train=True, combine=False)

    if cv_splits != None:
            train_datasets, val_datasets = k_fold_challenge_splitter(datasets, cv_splits, cv_fold)
    else:
        train_datasets = []
        val_datasets = []
        for ds in datasets:
            train, val = train_val_splits(ds, val_proportion, seed)
            train_datasets.append(train)
            val_datasets.append(val)

    train_ds = ConcatDataset(train_datasets)
    val_ds = ConcatDataset(val_datasets)
    
    if put_mask_in_label:
        train_ds = PutMaskInLabelDs3D(train_ds, False)
        val_ds = PutMaskInLabelDs3D(val_ds, False)

    if dims==3:
        if remove_mask_channel:
            train_ds = RemoveMaskDs(train_ds)
            val_ds = RemoveMaskDs(val_ds)
        return train_ds, val_ds

    train_ds_2D = WMH_Challenge_Dataset_2D(train_ds, transforms, remove_mask_channel)
    val_ds_2D = WMH_Challenge_Dataset_2D(val_ds, transforms, remove_mask_channel)

    return train_ds_2D, val_ds_2D

    