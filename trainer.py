import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    average_precision_score
)


class ImageTrainer:
    def __init__(self, model, device, lr=1e-4):
        self.model = model.to(device)
        self.device = device
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for imgs, targets in loader:
            imgs = imgs.to(self.device)
            targets = targets.float().to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(imgs)
            loss = self.criterion(preds, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()

        y_true = []
        y_pred = []
        total_loss = 0.0

        for imgs, targets in loader:
            imgs = imgs.to(self.device)
            targets = targets.float().to(self.device)

            preds = self.model(imgs)
            loss = self.criterion(preds, targets)

            total_loss += loss.item() * imgs.size(0)

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.clip(np.array(y_pred), 0, 1)

        return total_loss / len(loader.dataset), y_true, y_pred


class FusionTrainer:
    def __init__(self, model, device, lr=1e-4):
        self.model = model.to(device)
        self.device = device
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for imgs, tabs, targets in loader:
            imgs = imgs.to(self.device)
            tabs = tabs.float().to(self.device)
            targets = targets.float().to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(imgs, tabs)
            loss = self.criterion(preds, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()

        y_true = []
        y_pred = []
        total_loss = 0.0

        for imgs, tabs, targets in loader:
            imgs = imgs.to(self.device)
            tabs = tabs.float().to(self.device)
            targets = targets.float().to(self.device)

            preds = self.model(imgs, tabs)
            loss = self.criterion(preds, targets)

            total_loss += loss.item() * imgs.size(0)

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.clip(np.array(y_pred), 0, 1)

        return total_loss / len(loader.dataset), y_true, y_pred


def compute_metrics(y_true, y_pred, binary_thr=0.5):
    y_true_bin = (y_true >= binary_thr).astype(int)

    out = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
    }

    try:
        out["roc_auc"] = roc_auc_score(y_true_bin, y_pred)
    except Exception:
        out["roc_auc"] = np.nan

    try:
        out["pr_auc"] = average_precision_score(y_true_bin, y_pred)
    except Exception:
        out["pr_auc"] = np.nan

    return out
