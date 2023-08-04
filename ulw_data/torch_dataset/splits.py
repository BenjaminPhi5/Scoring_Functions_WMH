from torch.utils.data import random_split, Dataset
from torch import Generator
import numpy as np

def train_val_splits(dataset, val_prop, seed):
    # I think the sklearn version might be prefereable for determinism and things
    # but that involves fiddling with the dataset implementation I think....
    size = len(dataset)
    val_size = int(val_prop*size)
    train_size = size - val_size
    train, val = random_split(dataset, [train_size, val_size], generator=Generator().manual_seed(seed))
    return train, val
    
class FixedIndicesDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]
    
def k_fold_challenge_splitter(datasets, K, split=0):
    """
    hard-coded k-fold cross validation splitter for the challenge dataset
    K can be either 5, 6, 6-unequal, 10
    
    each training domain has 20 images.
    """
    
    # random shuffling applied to the indexes 0-19, once per dataset
    random_sample_0 = np.array([3, 11, 17, 14, 7, 6, 19, 2, 12, 16, 5, 10, 9, 1, 15, 18, 0, 8, 13, 4])
    random_sample_1 = np.array([0, 14, 7, 9, 8, 16, 6, 13, 5, 11, 2, 18, 17, 19, 3, 15, 4, 10, 1, 12])
    random_sample_2 = np.array([1, 8, 4, 13, 16, 9, 15, 6, 7, 3, 19, 18, 12, 5, 17, 10, 0, 2, 14, 11])
    
    dataset_indices = [random_sample_0, random_sample_1,random_sample_2]
    
    K = str(K)
    K_size = int(K[0:2])
    
    split_size = 20 // K_size
    val_indexes = (split_size * split, split_size * (split + 1))

    val_samples = [di[val_indexes[0]:val_indexes[1]] for di in dataset_indices]
    train_samples = [np.delete(di, np.arange(val_indexes[0], val_indexes[1], 1)) for di in dataset_indices]
    
    if K == '6-unequal':
        if split <= 1:
            val_samples[0] = np.array(list(val_samples[0]) + [dataset_indices[0][-1 -split % 2]])
            train_samples[0] = np.delete(train_samples[0], [-1 -split % 2])
        elif split <= 3:
            val_samples[1] = np.array(list(val_samples[1]) + [dataset_indices[1][-1 -split % 2]])
            train_samples[1] = np.delete(train_samples[1], [-1 -split % 2])
        else:
            val_samples[2] = np.array(list(val_samples[2]) + [dataset_indices[2][-1 -split % 2]])
            train_samples[2] = np.delete(train_samples[2], [-1 -split % 2])
        
    
    # print(val_samples)
    # print("--\n--")
    # print(train_samples)
    train_dss = [FixedIndicesDataset(ds, ids) for ds, ids in zip(datasets, train_samples)]
    val_dss = [FixedIndicesDataset(ds, ids) for ds, ids in zip(datasets, val_samples)]
    
    return train_dss, val_dss
    