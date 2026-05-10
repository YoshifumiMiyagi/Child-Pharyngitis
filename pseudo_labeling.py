import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import PharyngitisImageDataset


class ImagePseudoLabeler:
    def __init__(self, teacher_model, device, transform, batch_size=16):
        self.teacher_model = teacher_model
        self.device = device
        self.transform = transform
        self.batch_size = batch_size

    @torch.no_grad()
    def predict(self, df):
        ds = PharyngitisImageDataset(df, transform=self.transform)

        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

        self.teacher_model.eval()
        preds = []

        for imgs, _ in loader:
            imgs = imgs.to(self.device)
            out = self.teacher_model(imgs)
            preds.extend(out.cpu().numpy())

        return np.clip(np.array(preds).reshape(-1), 0, 1)

    def select_by_quantile(self, df, low_q=0.20, high_q=0.80):
        df = df.copy()
        preds = self.predict(df)

        df["teacher_pred_ratio"] = preds

        low_thr = df["teacher_pred_ratio"].quantile(low_q)
        high_thr = df["teacher_pred_ratio"].quantile(high_q)

        pseudo_neg = df[df["teacher_pred_ratio"] <= low_thr].copy()
        pseudo_pos = df[df["teacher_pred_ratio"] >= high_thr].copy()

        pseudo_df = pd.concat([pseudo_neg, pseudo_pos], axis=0).reset_index(drop=True)

        pseudo_df["bacterial_ratio"] = pseudo_df["teacher_pred_ratio"]
        pseudo_df["stratify_label"] = (
            pseudo_df["teacher_pred_ratio"] >= high_thr
        ).astype(int)
        pseudo_df["is_pseudo"] = 1

        return pseudo_df, low_thr, high_thr
