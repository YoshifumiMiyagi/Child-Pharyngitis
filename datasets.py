import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image


class PharyngitisImageDataset(Dataset):

    def __init__(
        self,
        df,
        transform=None,
        target_col="bacterial_ratio"
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img = Image.open(
            row["image_path"]
        ).convert("RGB")

        img = np.array(img)

        if self.transform:
            img = self.transform(
                image=img
            )["image"]

        y = torch.tensor(
            row[self.target_col],
            dtype=torch.float32
        )

        return img, y


class PharyngitisFusionDataset(Dataset):

    def __init__(
        self,
        df,
        tabular_cols,
        transform=None,
        target_col="bacterial_ratio"
    ):
        self.df = df.reset_index(drop=True)
        self.tabular_cols = tabular_cols
        self.transform = transform
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img = Image.open(
            row["image_path"]
        ).convert("RGB")

        img = np.array(img)

        if self.transform:
            img = self.transform(
                image=img
            )["image"]

        x_tab = torch.tensor(
            row[self.tabular_cols]
            .astype(float)
            .values,
            dtype=torch.float32
        )

        y = torch.tensor(
            row[self.target_col],
            dtype=torch.float32
        )

        return img, x_tab, y
