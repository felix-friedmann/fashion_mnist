from torch.utils.data import Dataset
from src.config import *

class AugmentDataset(Dataset):
    """
    Augment dataset.
    """
    def __init__(self, base_dataset):
        self.dataset = base_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        target = int(target)

        # if target in CONFUSED_CLASSES:
        #     data = STRONG_TRANSFORM(data)
        # else:
        data = BASE_TRANSFORM(data)

        return data, target