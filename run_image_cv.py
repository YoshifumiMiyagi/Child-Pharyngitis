import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from datasets import PharyngitisImageDataset
from models import ResNet18Regressor
from trainer import ImageTrainer, compute_metrics


def run_image_cv(
    df,
    train_tfms,
    valid_tfms,
    batch_size=16,
    epochs=10,
    lr=1e-4,
    n_splits=5,
    device=None,
    output_csv="oof_image.csv"
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    y_strat = df["stratify_label"].values
    X_dummy = np.zeros(len(df))

    oof_pred = np.zeros(len(df), dtype=np.float32)
    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_dummy, y_strat), 1):
        print(f"\n===== Fold {fold} =====")

        train_df = df.iloc[tr_idx].reset_index(drop=True)
        valid_df = df.iloc[va_idx].reset_index(drop=True)

        train_ds = PharyngitisImageDataset(train_df, transform=train_tfms)
        valid_ds = PharyngitisImageDataset(valid_df, transform=valid_tfms)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        model = ResNet18Regressor(pretrained=True)
        trainer = ImageTrainer(model, device=device, lr=lr)

        best_mae = np.inf
        best_state = None

        for epoch in range(epochs):
            train_loss = trainer.train_one_epoch(train_loader)
            valid_loss, y_true, y_pred = trainer.predict(valid_loader)

            metrics = compute_metrics(y_true, y_pred)

            print(
                f"fold={fold} epoch={epoch+1:02d}/{epochs} "
                f"train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} "
                f"mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f} "
                f"roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f}"
            )

            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_state = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}

        trainer.model.load_state_dict(best_state)

        valid_loss, y_true, y_pred = trainer.predict(valid_loader)
        oof_pred[va_idx] = y_pred

        metrics = compute_metrics(y_true, y_pred)
        metrics["fold"] = fold
        metrics["n_train"] = len(train_df)
        metrics["n_valid"] = len(valid_df)
        fold_results.append(metrics)

    result_df = pd.DataFrame(fold_results)

    y_all = df["bacterial_ratio"].values
    overall = compute_metrics(y_all, oof_pred)

    print("\n=== CV summary ===")
    print(result_df[["mae", "rmse", "roc_auc", "pr_auc"]].agg(["mean", "std"]))

    print("\n=== OOF overall ===")
    print(overall)

    df_out = df.copy()
    df_out["oof_pred_image"] = oof_pred
    df_out.to_csv(output_csv, index=False)

    result_df.to_csv("image_5cv_results.csv", index=False)

    return df_out, result_df
