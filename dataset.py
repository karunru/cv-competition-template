import numpy as np
import torch
from torch.utils.data import Dataset


class SetiDataset(Dataset):
    def __init__(self, mode, X, y=None, transform=None):
        self.mode = mode
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = np.vstack(np.load(self.X[idx])[::2]).astype(float)

        # https://www.kaggle.com/c/seti-breakthrough-listen/discussion/248194
        for i in range(image.shape[0]):
            image[i] -= image[i].mean()
            image[i] /= image[i].std()

        image = self.transform(image=image)["image"]

        label = (
            torch.tensor(self.y[idx]).float().unsqueeze(-1)
            if self.y is not None
            else None
        )
        if label is not None:
            return image, label
        else:
            return image


class SetiEachDataset(Dataset):
    def __init__(self, mode, X, ids, y=None, transform=None):
        self.mode = mode
        self.X = X
        self.ids = ids
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = np.vstack(np.load(self.X[idx])[self.ids[idx]]).astype(float)

        # https://www.kaggle.com/c/seti-breakthrough-listen/discussion/248194
        image -= image.mean()
        image /= image.std()

        image = self.transform(image=image)["image"]

        label = (
            torch.tensor(self.y[idx]).float().unsqueeze(-1)
            if self.y is not None
            else None
        )
        if label is not None:
            return image, label
        else:
            return image
