# =========================================================
# dataset.py
# =========================================================

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# =========================================================
# dataset
# =========================================================

class PharyngitisDataset(Dataset):

    def __init__(
        self,
        cfg,
        df,
        mode="train",
    ):

        self.cfg = cfg
        self.df = df
        self.mode = mode

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = row["image_path"]

        img = cv2.imread(img_path)

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img = cv2.resize(
            img,
            (self.cfg.img_size, self.cfg.img_size)
        )

        img = img.astype(np.float32) / 255.0

        img = np.transpose(img, (2, 0, 1))

        img = torch.tensor(img).float()

        label = torch.tensor(
            row[self.cfg.target_col]
        ).float()

        # =============================================
        # image only
        # =============================================

        if not self.cfg.use_table:

            return img, label

        # =============================================
        # fusion
        # =============================================

        table = row[list(self.cfg.table_cols)].values
        table = torch.tensor(table).float()

        return (img, table), label


# =========================================================
# dataloader
# =========================================================

def build_dataloader(
    cfg,
    df,
    mode="train",
):

    dataset = PharyngitisDataset(
        cfg=cfg,
        df=df,
        mode=mode,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(mode == "train"),
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return loader
