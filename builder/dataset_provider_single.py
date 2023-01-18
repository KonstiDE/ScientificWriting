import os
import random

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class NrwDataSet(Dataset):
    def __init__(self, npz_dir, percentage_load, below_m):
        files = os.listdir(npz_dir)

        self.dataset = []
        self.below_m = below_m

        for u in range(int(percentage_load * len(files))):
            file = random.choice(files)
            self.dataset.append(os.path.join(npz_dir, file))
            files.remove(file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        dataframepath = self.dataset[index]
        dataframe = np.load(dataframepath, allow_pickle=True)

        return dataframepath, \
            torch.Tensor(
                np.stack((
                    dataframe["red"],
                    dataframe["green"],
                    dataframe["blue"],
                    dataframe["nir"]
                )).astype(dtype=np.int32)), \
            torch.Tensor(dataframe["dsm_below" + str(self.below_m)]), \
            torch.Tensor(dataframe["dsm_og"])


def get_loader(npz_dir, batch_size, percentage_load, below_m, num_workers=2, pin_memory=True, shuffle=True):
    train_ds = NrwDataSet(npz_dir, percentage_load, below_m)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle
    )
    return train_loader


def get_dataset(npz_dir, percentage_load, below_m):
    return NrwDataSet(npz_dir, percentage_load, below_m)
