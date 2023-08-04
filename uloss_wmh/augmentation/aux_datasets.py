from torch.utils.data import Dataset

class AugmentAndExtractDataset(Dataset):
    def __init__(self, base_dataset, transforms=None, keys=['image', 'label']):
        """
        applies augmentations, and returns specified keys from the map that the base_dataset comes with
        """
        self.base_dataset = base_dataset
        self.transforms = transforms
        self.keys = keys

    def __getitem__(self, idx):
        data = self.base_dataset[idx]

        if self.transforms:
            data = self.transforms(data)

        return [data[key] for key in self.keys]

    def __len__(self):
        return len(self.base_dataset)