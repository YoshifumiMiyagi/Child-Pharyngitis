#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pseudo-label training script for pharyngitis image classification/regression.

This script supports:
1. Supervised training on labeled pediatric/adult data
2. Teacher-model inference for pseudo-label generation
3. Confidence filtering of pseudo-labeled samples
4. Student training using real + pseudo-labeled data

Expected CSV columns:
- image_path: path to image file
- bacterial_ratio: target value between 0 and 1 for labeled data

Optional columns:
- patient_ID
- age
- group
"""

import os
import random
import argparse
from collections import OrderedDict
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score


# ============================================================
# Utility
# ============================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Dataset
# ============================================================
class PharyngitisDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        target_col: Optional[str] = "bacterial_ratio",
        return_target: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.target_col = target_col
        self.return_target = return_target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.return_target:
            y = torch.tensor(row[self.target_col], dtype=torch.float32)
            return img, y

        return img


# ============================================================
# Transform
# ============================================================
class TransformFactory:
    def __init__(self, img_size: int = 224):
        self.img_size = img_size

    def train(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])

    def valid(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])


# ============================================================
# Model
# ============================================================
class PharyngitisRegressor(nn.Module):
    def __init__(self, model_name: str = "resnet10t", pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1,
        )

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x.squeeze(1)


# ============================================================
# Trainer
# ============================================================
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: str,
        lr: float = 1e-4,
        pseudo_weight: float = 0.3,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction="none")
        self.pseudo_weight = pseudo_weight

    def train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_n = 0

        for batch in loader:
            if len(batch) == 3:
                imgs, targets, is_pseudo = batch
                is_pseudo = is_pseudo.to(self.device).float()
            else:
                imgs, targets = batch
                is_pseudo = torch.zeros_like(targets)

            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            is_pseudo = is_pseudo.to(self.device)

            preds = self.model(imgs)
            loss_each = self.criterion(preds, targets)

            weights = torch.where(
                is_pseudo > 0,
                torch.full_like(loss_each, self.pseudo_weight),
                torch.ones_like(loss_each),
            )

            loss = (loss_each * weights).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_n += imgs.size(0)

        return total_loss / total_n

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        total_n = 0
        all_targets = []
        all_preds = []

        for imgs, targets in loader:
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)

            preds = self.model(imgs)
            loss_each = self.criterion(preds, targets)
            loss = loss_each.mean()

            total_loss += loss.item() * imgs.size(0)
            total_n += imgs.size(0)

            all_targets.extend(targets.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())

        return total_loss / total_n, np.array(all_targets), np.array(all_preds)


class PseudoAwareDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        y = torch.tensor(row["bacterial_ratio"], dtype=torch.float32)
        is_pseudo = torch.tensor(row.get("is_pseudo", 0), dtype=torch.float32)

        return img, y, is_pseudo


# ============================================================
# Pseudo Labeler
# ============================================================
class PseudoLabeler:
    def __init__(
        self,
        model: nn.Module,
        device: str,
        transform,
        batch_size: int = 16,
        num_workers: int = 2,
    ):
        self.model = model.to(device)
        self.device = device
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ds = PharyngitisDataset(
            df,
            transform=self.transform,
            target_col=None,
            return_target=False,
        )
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.model.eval()
        preds = []

        for imgs in loader:
            imgs = imgs.to(self.device)
            p = self.model(imgs)
            preds.extend(p.detach().cpu().numpy())

        return np.array(preds)

    def create_pseudo_dataframe(
        self,
        unlabeled_df: pd.DataFrame,
        low_th: float = 0.10,
        high_th: float = 0.90,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pseudo_df = unlabeled_df.copy()
        pseudo_df = pseudo_df[pseudo_df["image_path"].notna()].reset_index(drop=True)

        preds = self.predict(pseudo_df)

        pseudo_df["pseudo_bacterial_ratio"] = preds
        pseudo_df["pseudo_label_binary"] = (preds >= 0.5).astype(int)

        pseudo_df["pseudo_confident"] = (
            (pseudo_df["pseudo_bacterial_ratio"] <= low_th) |
            (pseudo_df["pseudo_bacterial_ratio"] >= high_th)
        )

        pseudo_keep = pseudo_df[pseudo_df["pseudo_confident"]].copy()
        pseudo_keep["bacterial_ratio"] = pseudo_keep["pseudo_bacterial_ratio"]
        pseudo_keep["stratify_label"] = (pseudo_keep["bacterial_ratio"] >= 0.5).astype(int)
        pseudo_keep["is_pseudo"] = 1

        return pseudo_df, pseudo_keep.reset_index(drop=True)


# ============================================================
# Cross-validation
# ============================================================
class CrossValidator:
    def __init__(
        self,
        model_name: str,
        img_size: int,
        batch_size: int,
        epochs: int,
        lr: float,
        n_splits: int,
        device: str,
        output_dir: str,
        pseudo_weight: float = 0.3,
        num_workers: int = 2,
    ):
        self.model_name = model_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.n_splits = n_splits
        self.device = device
        self.output_dir = output_dir
        self.pseudo_weight = pseudo_weight
        self.num_workers = num_workers

        os.makedirs(self.output_dir, exist_ok=True)
        self.tfms = TransformFactory(img_size)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df[df["image_path"].notna()].reset_index(drop=True)

        if "stratify_label" not in df.columns:
            df["stratify_label"] = (df["bacterial_ratio"] >= 0.5).astype(int)

        if "is_pseudo" not in df.columns:
            df["is_pseudo"] = 0

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=42,
        )

        X_dummy = np.zeros(len(df))
        y_strat = df["stratify_label"].values

        oof_pred = np.zeros(len(df), dtype=np.float32)
        fold_results = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_dummy, y_strat), 1):
            print("\n" + "=" * 80)
            print(f"FOLD {fold}")
            print("=" * 80)

            train_df = df.iloc[tr_idx].reset_index(drop=True)
            valid_df = df.iloc[va_idx].reset_index(drop=True)

            print("train:", len(train_df), "valid:", len(valid_df))
            print("pseudo in train:", int(train_df["is_pseudo"].sum()))

            train_ds = PseudoAwareDataset(train_df, transform=self.tfms.train())
            valid_ds = PharyngitisDataset(valid_df, transform=self.tfms.valid())

            train_loader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            valid_loader = DataLoader(
                valid_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

            model = PharyngitisRegressor(
                model_name=self.model_name,
                pretrained=True,
            )

            trainer = Trainer(
                model=model,
                device=self.device,
                lr=self.lr,
                pseudo_weight=self.pseudo_weight,
            )

            best_mae = np.inf
            best_state = None

            for epoch in range(self.epochs):
                train_loss = trainer.train_one_epoch(train_loader)
                valid_loss, y_true, y_pred = trainer.predict(valid_loader)

                mae = mean_absolute_error(y_true, y_pred)
                rmse = mean_squared_error(y_true, y_pred) ** 0.5

                y_true_bin = (y_true >= 0.5).astype(int)
                try:
                    auc = roc_auc_score(y_true_bin, y_pred)
                except Exception:
                    auc = np.nan

                print(
                    f"fold={fold} epoch={epoch+1:02d}/{self.epochs} "
                    f"train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} "
                    f"mae={mae:.4f} rmse={rmse:.4f} auc={auc:.4f}"
                )

                if mae < best_mae:
                    best_mae = mae
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in trainer.model.state_dict().items()
                    }

            model_path = os.path.join(self.output_dir, f"best_model_fold{fold}.pth")
            torch.save(best_state, model_path)

            trainer.model.load_state_dict(best_state)
            valid_loss, y_true, y_pred = trainer.predict(valid_loader)

            oof_pred[va_idx] = y_pred

            fold_mae = mean_absolute_error(y_true, y_pred)
            fold_rmse = mean_squared_error(y_true, y_pred) ** 0.5

            y_true_bin = (y_true >= 0.5).astype(int)
            try:
                fold_auc = roc_auc_score(y_true_bin, y_pred)
            except Exception:
                fold_auc = np.nan

            fold_results.append({
                "fold": fold,
                "n_train": len(train_df),
                "n_valid": len(valid_df),
                "n_pseudo_train": int(train_df["is_pseudo"].sum()),
                "mae": fold_mae,
                "rmse": fold_rmse,
                "auc_binary_ref": fold_auc,
            })

        result_df = pd.DataFrame(fold_results)
        result_df.to_csv(os.path.join(self.output_dir, "cv_results.csv"), index=False)

        df["oof_pred_ratio"] = oof_pred
        df.to_csv(os.path.join(self.output_dir, "oof_predictions.csv"), index=False)

        self.average_fold_weights()

        print("\n=== CV summary ===")
        print(result_df[["mae", "rmse", "auc_binary_ref"]].agg(["mean", "std"]))

        y_true_all = df["bacterial_ratio"].values
        y_pred_all = oof_pred

        print("\n=== OOF overall ===")
        print("OOF MAE :", mean_absolute_error(y_true_all, y_pred_all))
        print("OOF RMSE:", mean_squared_error(y_true_all, y_pred_all) ** 0.5)

        try:
            print(
                "OOF AUC(binary ref):",
                roc_auc_score((y_true_all >= 0.5).astype(int), y_pred_all),
            )
        except Exception:
            print("OOF AUC(binary ref): nan")

        return result_df

    def average_fold_weights(self):
        paths = [
            os.path.join(self.output_dir, f"best_model_fold{i}.pth")
            for i in range(1, self.n_splits + 1)
        ]

        avg_state = None

        for p in paths:
            state = torch.load(p, map_location="cpu")

            if avg_state is None:
                avg_state = OrderedDict()
                for k, v in state.items():
                    avg_state[k] = v.clone().float()
            else:
                for k, v in state.items():
                    avg_state[k] += v.float()

        for k in avg_state:
            avg_state[k] /= len(paths)

        save_path = os.path.join(self.output_dir, "best_model_fold_avg.pth")
        torch.save(avg_state, save_path)
        print(f"saved: {save_path}")


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "pseudo", "train_with_pseudo"])

    parser.add_argument("--labeled_csv", type=str, default=None)
    parser.add_argument("--unlabeled_csv", type=str, default=None)
    parser.add_argument("--teacher_weight", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--model_name", type=str, default="resnet10t")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--low_th", type=float, default=0.10)
    parser.add_argument("--high_th", type=float, default=0.90)
    parser.add_argument("--pseudo_weight", type=float, default=0.3)

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "train":
        if args.labeled_csv is None:
            raise ValueError("--labeled_csv is required for train mode")

        df = pd.read_csv(args.labeled_csv)

        cv = CrossValidator(
            model_name=args.model_name,
            img_size=args.img_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            n_splits=args.n_splits,
            device=device,
            output_dir=args.output_dir,
            pseudo_weight=args.pseudo_weight,
            num_workers=args.num_workers,
        )
        cv.run(df)

    elif args.mode == "pseudo":
        if args.unlabeled_csv is None:
            raise ValueError("--unlabeled_csv is required for pseudo mode")
        if args.teacher_weight is None:
            raise ValueError("--teacher_weight is required for pseudo mode")

        unlabeled_df = pd.read_csv(args.unlabeled_csv)

        tfms = TransformFactory(args.img_size)
        teacher = PharyngitisRegressor(
            model_name=args.model_name,
            pretrained=False,
        )
        teacher.load_state_dict(torch.load(args.teacher_weight, map_location=device))

        labeler = PseudoLabeler(
            model=teacher,
            device=device,
            transform=tfms.valid(),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        pseudo_all, pseudo_keep = labeler.create_pseudo_dataframe(
            unlabeled_df,
            low_th=args.low_th,
            high_th=args.high_th,
        )

        pseudo_all_path = os.path.join(args.output_dir, "pseudo_all_predictions.csv")
        pseudo_keep_path = os.path.join(args.output_dir, "pseudo_confident_used.csv")

        pseudo_all.to_csv(pseudo_all_path, index=False)
        pseudo_keep.to_csv(pseudo_keep_path, index=False)

        print(f"pseudo all: {len(pseudo_all)}")
        print(f"pseudo kept: {len(pseudo_keep)}")
        print(f"saved: {pseudo_all_path}")
        print(f"saved: {pseudo_keep_path}")

    elif args.mode == "train_with_pseudo":
        if args.labeled_csv is None:
            raise ValueError("--labeled_csv is required for train_with_pseudo mode")
        if args.unlabeled_csv is None:
            raise ValueError("--unlabeled_csv is required for train_with_pseudo mode")
        if args.teacher_weight is None:
            raise ValueError("--teacher_weight is required for train_with_pseudo mode")

        labeled_df = pd.read_csv(args.labeled_csv)
        labeled_df["is_pseudo"] = 0

        unlabeled_df = pd.read_csv(args.unlabeled_csv)

        tfms = TransformFactory(args.img_size)
        teacher = PharyngitisRegressor(
            model_name=args.model_name,
            pretrained=False,
        )
        teacher.load_state_dict(torch.load(args.teacher_weight, map_location=device))

        labeler = PseudoLabeler(
            model=teacher,
            device=device,
            transform=tfms.valid(),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        pseudo_all, pseudo_keep = labeler.create_pseudo_dataframe(
            unlabeled_df,
            low_th=args.low_th,
            high_th=args.high_th,
        )

        pseudo_all.to_csv(
            os.path.join(args.output_dir, "pseudo_all_predictions.csv"),
            index=False,
        )
        pseudo_keep.to_csv(
            os.path.join(args.output_dir, "pseudo_confident_used.csv"),
            index=False,
        )

        train_df = pd.concat([labeled_df, pseudo_keep], axis=0).reset_index(drop=True)
        train_df.to_csv(
            os.path.join(args.output_dir, "train_with_pseudo.csv"),
            index=False,
        )

        print("real labeled:", len(labeled_df))
        print("pseudo used:", len(pseudo_keep))
        print("combined:", len(train_df))

        cv = CrossValidator(
            model_name=args.model_name,
            img_size=args.img_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            n_splits=args.n_splits,
            device=device,
            output_dir=args.output_dir,
            pseudo_weight=args.pseudo_weight,
            num_workers=args.num_workers,
        )
        cv.run(train_df)


if __name__ == "__main__":
    main()
