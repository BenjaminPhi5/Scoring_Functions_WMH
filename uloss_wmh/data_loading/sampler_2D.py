import torch
from torch.utils.data import Sampler
import numpy as np

class BinaryTargetClassBatchSampler(Sampler):
    def __init__(self, dataset, target_class, target_class_proportion, batch_size, num_iterations):
        self.dataset = dataset
        self.target_class = target_class
        self.batch_size = batch_size
        assert 0 <= target_class_proportion <= 1
        self.targets_per_batch = max(int(np.round(target_class_proportion * batch_size)), 1)
        self.remaining_per_batch = batch_size - self.targets_per_batch
        self.num_iterations = num_iterations

        # record the indices of instances containing the target class
        self.target_indices = []
        for i, (_, y) in enumerate(self.dataset): # assumes dataset outputs x, y pairs (which mine can do after being wrapped in the AugmentAndExtractDataset)
            if y.shape[0] == 1:
                if (y == target_class).max() > 0:
                    self.target_indices.append(i)
            else: # case that the labels are one-hot-encoded
                if (y[target_class] == 1).max() > 0:
                    self.target_indices.append(i)
                    
        self.all_indices = torch.arange(0, len(dataset), 1).tolist()

        print(len(self.target_indices))

    def __iter__(self):
        while True:
            # shuffle the target indices
            target_indices = torch.randperm(len(self.target_indices)).tolist()

            # sample the required number of target indices without replacement
            target_batch = target_indices[:self.targets_per_batch]

            # remove the selected target indices from the all_indices vector
            remaining_indices = list(set(self.all_indices) - set(target_batch))

            # select the required number of remaining indices to fill the batch, without replacement
            np.random.shuffle(remaining_indices)
            remaining_batch = remaining_indices[:self.remaining_per_batch]

            # construct the batch such that target and remaining examples are randomly shuffled together
            batch = target_batch + remaining_batch
            random_indices = torch.randperm(len(batch)).tolist()
            batch = [batch[i] for i in random_indices]

            yield batch
            

    def __len__(self):
        return self.num_iterations