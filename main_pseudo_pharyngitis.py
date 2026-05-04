# main.py

import os
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score


# =========================
# Seed
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Dataset
# =========================
class PharyngitisFusionDataset(Dataset):
    def __init__(self, df, tabular_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.tabular_cols = tabular_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        x_tab = torch.tensor(
            row[self.tabular_cols].astype(float).values,
            dtype=torch.float32
        )

        y = torch.tensor(row["bacterial_ratio"], dtype=torch.float32)

        return img, x_tab, y


# =========================
# Model
# =========================
class ResNet18FusionRegressor(nn.Module):
    def __init__(self, n_tab, pretrained=True):
        super().__init__()

        base = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        in_features = base.fc.in_features
        base.fc = nn.Identity()

        self.backbone = base

        self.tab = nn.Sequential(
            nn.Linear(n_tab, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.head = nn.Sequential(
            nn.Linear(in_features + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, img, x_tab):
        img_feat = self.backbone(img)
        tab_feat = self.tab(x_tab)

        x = torch.cat([img_feat, tab_feat], dim=1)
        x = self.head(x)

        return torch.sigmoid(x).squeeze(1)


# =========================
# Train / Valid
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for imgs, x_tab, targets in loader:
        imgs = imgs.to(device)
        x_tab = x_tab.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(imgs, x_tab)
        loss = criterion(preds, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_valid(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    all_targets = []
    all_preds = []

    for imgs, x_tab, targets in loader:
        imgs = imgs.to(device)
        x_tab = x_tab.to(device)
        targets = targets.to(device)

        preds = model(imgs, x_tab)
        loss = criterion(preds, targets)

        total_loss += loss.item() * imgs.size(0)

        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    return (
        total_loss / len(loader.dataset),
        np.array(all_targets),
        np.array(all_preds)
    )


# =========================
# Grad-CAM++
# =========================
class GradCAMPlusPlusFusion:
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self.activations = None
        self.gradients = None

        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate(self, imgs, x_tab):
        self.model.zero_grad()

        preds = self.model(imgs, x_tab)
        score = preds.sum()
        score.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations

        grads2 = grads ** 2
        grads3 = grads2 * grads

        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        alpha = grads2 / (2 * grads2 + sum_acts * grads3 + 1e-8)

        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=imgs.shape[-2:],
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

        cam_min = cam.flatten(1).min(dim=1)[0].view(-1, 1, 1)
        cam_max = cam.flatten(1).max(dim=1)[0].view(-1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach().cpu(), preds.detach().cpu()


def contrast_stretch(img, low=2, high=98):
    lo = np.percentile(img, low)
    hi = np.percentile(img, high)
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)


def overlay_cam(img, cam, alpha=0.35):
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()

    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    img = np.clip(img, 0, 1)

    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    overlay = (1 - alpha) * img + alpha * heatmap
    return np.clip(overlay, 0, 1)


def make_mean_cam_figure(
    model,
    loader,
    device,
    out_path,
    prob_low=0.1,
    prob_high=0.9,
    alpha=0.35
):
    model.eval()

    target_layer = model.backbone.layer4[-1].conv2
    cam_analyzer = GradCAMPlusPlusFusion(model, target_layer, device)

    imgs_all = []
    cams_all = []

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for imgs, x_tab, targets in loader:
        imgs = imgs.to(device)
        x_tab = x_tab.to(device)

        with torch.no_grad():
            probs = model(imgs, x_tab)

        mask = (probs < prob_low) | (probs > prob_high)

        if mask.sum().item() == 0:
            continue

        imgs_f = imgs[mask]
        x_tab_f = x_tab[mask]

        cams, probs_f = cam_analyzer.generate(imgs_f, x_tab_f)

        imgs_all.append(imgs_f.detach().cpu())
        cams_all.append(cams.detach().cpu())

    cam_analyzer.remove_hooks()

    if len(imgs_all) == 0:
        print("No samples passed confidence filter.")
        return

    imgs_all = torch.cat(imgs_all, dim=0)
    cams_all = torch.cat(cams_all, dim=0)

    print("Grad-CAM N used:", len(imgs_all))

    # median image
    mean_img = imgs_all.median(dim=0).values
    mean_img = mean_img * std + mean
    mean_img = mean_img.clamp(0, 1)

    img_np = mean_img.numpy().transpose(1, 2, 0)
    img_np = contrast_stretch(img_np, low=2, high=98)

    mean_cam = cams_all.mean(dim=0).numpy()
    mean_cam = (mean_cam - mean_cam.min()) / (mean_cam.max() - mean_cam.min() + 1e-6)

    overlay = overlay_cam(img_np, mean_cam, alpha=alpha)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_np)
    axes[0].set_title("Median image")
    axes[0].axis("off")

    axes[1].imshow(mean_cam, cmap="jet")
    axes[1].set_title("Mean activation Grad-CAM++")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


# =========================
# Main
# =========================
def main(args):
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    tab_cols = args.tab_cols.split(",")

    train_tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    valid_tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    df = pd.read_csv(args.csv_path)

    # sex preprocessing
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].replace({
            "Male": 0,
            "Female": 1,
            "M": 0,
            "F": 1,
            "male": 0,
            "female": 1
        })

    df[tab_cols] = df[tab_cols].apply(pd.to_numeric, errors="coerce")
    df[tab_cols] = df[tab_cols].fillna(0)

    real_df = df[df["is_pseudo"] == 0].reset_index(drop=True)
    pseudo_df = df[df["is_pseudo"] == 1].reset_index(drop=True)

    print("real:", len(real_df))
    print("pseudo:", len(pseudo_df))
    print("real label:")
    print(real_df["stratify_label"].value_counts())

    skf = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed
    )

    oof_pred = np.zeros(len(real_df), dtype=np.float32)
    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(
        skf.split(np.zeros(len(real_df)), real_df["stratify_label"]),
        1
    ):
        print("\n" + "=" * 80)
        print(f"FOLD {fold}")
        print("=" * 80)

        real_train_df = real_df.iloc[tr_idx].reset_index(drop=True)
        valid_df = real_df.iloc[va_idx].reset_index(drop=True)

        if args.use_pseudo:
            train_df = pd.concat(
                [real_train_df, pseudo_df],
                axis=0
            ).reset_index(drop=True)
        else:
            train_df = real_train_df.copy()

        print("train real:", len(real_train_df))
        print("train pseudo:", len(pseudo_df) if args.use_pseudo else 0)
        print("valid real:", len(valid_df))
        print("train label:")
        print(train_df["stratify_label"].value_counts())
        print("valid label:")
        print(valid_df["stratify_label"].value_counts())

        train_ds = PharyngitisFusionDataset(
            train_df,
            tabular_cols=tab_cols,
            transform=train_tfms
        )

        valid_ds = PharyngitisFusionDataset(
            valid_df,
            tabular_cols=tab_cols,
            transform=valid_tfms
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        model = ResNet18FusionRegressor(
            n_tab=len(tab_cols),
            pretrained=True
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_mae = np.inf
        best_state = None

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device
            )

            valid_loss, y_true, y_pred = predict_valid(
                model,
                valid_loader,
                criterion,
                device
            )

            mae = mean_absolute_error(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred) ** 0.5

            y_true_bin = (y_true >= 0.5).astype(int)
            try:
                auc = roc_auc_score(y_true_bin, y_pred)
            except Exception:
                auc = np.nan

            print(
                f"fold={fold} epoch={epoch+1:02d}/{args.epochs} "
                f"train_loss={train_loss:.4f} "
                f"valid_loss={valid_loss:.4f} "
                f"mae={mae:.4f} "
                f"rmse={rmse:.4f} "
                f"auc={auc:.4f}"
            )

            if mae < best_mae:
                best_mae = mae
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

        model.load_state_dict(best_state)

        model_path = os.path.join(args.out_dir, f"best_model_fold{fold}.pth")
        torch.save(best_state, model_path)

        valid_loss, y_true, y_pred = predict_valid(
            model,
            valid_loader,
            criterion,
            device
        )

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
            "n_train_real": len(real_train_df),
            "n_train_pseudo": len(pseudo_df) if args.use_pseudo else 0,
            "n_valid_real": len(valid_df),
            "mae": fold_mae,
            "rmse": fold_rmse,
            "auc_binary_ref": fold_auc
        })

        if args.save_cam:
            cam_path = os.path.join(args.out_dir, f"gradcampp_fold{fold}.png")
            make_mean_cam_figure(
                model=model,
                loader=valid_loader,
                device=device,
                out_path=cam_path,
                prob_low=args.cam_prob_low,
                prob_high=args.cam_prob_high,
                alpha=args.cam_alpha
            )

    result_df = pd.DataFrame(fold_results)
    result_path = os.path.join(args.out_dir, "cv_results.csv")
    result_df.to_csv(result_path, index=False)

    real_df["oof_pred_ratio"] = oof_pred
    oof_path = os.path.join(args.out_dir, "oof_real_predictions.csv")
    real_df.to_csv(oof_path, index=False)

    print("\n=== Fold results ===")
    print(result_df)

    print("\n=== CV summary on real validation only ===")
    print(result_df[["mae", "rmse", "auc_binary_ref"]].agg(["mean", "std"]))

    print("Saved:", result_path)
    print("Saved:", oof_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, default="train_with_pseudo.csv")
    parser.add_argument("--out_dir", type=str, default="./outputs")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--tab_cols", type=str, default="Age,Sex")
    parser.add_argument("--use_pseudo", action="store_true")

    parser.add_argument("--save_cam", action="store_true")
    parser.add_argument("--cam_prob_low", type=float, default=0.1)
    parser.add_argument("--cam_prob_high", type=float, default=0.9)
    parser.add_argument("--cam_alpha", type=float, default=0.35)

    args = parser.parse_args()
    main(args)
