import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from datasets import PharyngitisImageDataset
from models import ResNet18Regressor
from trainer import ImageTrainer, compute_metrics
from pseudo_labeling import ImagePseudoLabeler


def run_image_pseudo_cv(
    real_df,
    adult_df,
    train_tfms,
    valid_tfms,
    batch_size=16,
    epochs=10,
    lr=1e-4,
    n_splits=5,
    low_q=0.20,
    high_q=0.80,
    device=None,
    output_csv="oof_image_pseudo.csv"
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    y_strat = real_df["stratify_label"].values
    X_dummy = np.zeros(len(real_df))

    oof_pred = np.zeros(len(real_df), dtype=np.float32)
    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_dummy, y_strat), 1):
        print(f"\n===== Fold {fold} =====")

        real_train_df = real_df.iloc[tr_idx].copy().reset_index(drop=True)
        valid_df = real_df.iloc[va_idx].copy().reset_index(drop=True)
        adult_fold_df = adult_df.copy().reset_index(drop=True)

        # teacher
        teacher_train_ds = PharyngitisImageDataset(real_train_df, transform=train_tfms)
        teacher_valid_ds = PharyngitisImageDataset(valid_df, transform=valid_tfms)

        teacher_train_loader = DataLoader(teacher_train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        teacher_valid_loader = DataLoader(teacher_valid_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        teacher = ResNet18Regressor(pretrained=True)
        teacher_trainer = ImageTrainer(teacher, device=device, lr=lr)

        best_mae = np.inf
        best_state = None

        for epoch in range(epochs):
            teacher_trainer.train_one_epoch(teacher_train_loader)
            _, y_true, y_pred = teacher_trainer.predict(teacher_valid_loader)

            metrics = compute_metrics(y_true, y_pred)

            print(
                f"[Teacher] fold={fold} epoch={epoch+1:02d}/{epochs} "
                f"mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f} "
                f"roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f}"
            )

            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_state = {k: v.cpu().clone() for k, v in teacher_trainer.model.state_dict().items()}

        teacher_trainer.model.load_state_dict(best_state)

        # pseudo-label adult images
        pseudo_labeler = ImagePseudoLabeler(
            teacher_model=teacher_trainer.model,
            device=device,
            transform=valid_tfms,
            batch_size=batch_size
        )

        pseudo_df, low_thr, high_thr = pseudo_labeler.select_by_quantile(
            adult_fold_df,
            low_q=low_q,
            high_q=high_q
        )

        print("adult total:", len(adult_fold_df))
        print("pseudo selected:", len(pseudo_df))
        print("pseudo labels:", pseudo_df["stratify_label"].value_counts().to_dict())
        print("low_thr:", low_thr, "high_thr:", high_thr)

        # student
        student_train_df = pd.concat(
            [real_train_df, pseudo_df],
            axis=0
        ).reset_index(drop=True)

        student_train_ds = PharyngitisImageDataset(student_train_df, transform=train_tfms)
        student_valid_ds = PharyngitisImageDataset(valid_df, transform=valid_tfms)

        student_train_loader = DataLoader(student_train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        student_valid_loader = DataLoader(student_valid_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        student = ResNet18Regressor(pretrained=True)
        student_trainer = ImageTrainer(student, device=device, lr=lr)

        best_mae = np.inf
        best_state = None

        for epoch in range(epochs):
            student_trainer.train_one_epoch(student_train_loader)
            _, y_true, y_pred = student_trainer.predict(student_valid_loader)

            metrics = compute_metrics(y_true, y_pred)

            print(
                f"[Student] fold={fold} epoch={epoch+1:02d}/{epochs} "
                f"mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f} "
                f"roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f}"
            )

            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_state = {k: v.cpu().clone() for k, v in student_trainer.model.state_dict().items()}

        student_trainer.model.load_state_dict(best_state)

        _, y_true, y_pred = student_trainer.predict(student_valid_loader)
        oof_pred[va_idx] = y_pred

        metrics = compute_metrics(y_true, y_pred)
        metrics["fold"] = fold
        metrics["n_train_real"] = len(real_train_df)
        metrics["n_adult_total"] = len(adult_fold_df)
        metrics["n_pseudo_selected"] = len(pseudo_df)
        metrics["n_train_total"] = len(student_train_df)
        metrics["n_valid_real"] = len(valid_df)

        fold_results.append(metrics)

    result_df = pd.DataFrame(fold_results)

    y_all = real_df["bacterial_ratio"].values
    overall = compute_metrics(y_all, oof_pred)

    print("\n=== CV summary ===")
    print(result_df[["mae", "rmse", "roc_auc", "pr_auc"]].agg(["mean", "std"]))

    print("\n=== OOF overall ===")
    print(overall)

    df_out = real_df.copy()
    df_out["oof_pred_image_pseudo"] = oof_pred
    df_out.to_csv(output_csv, index=False)

    result_df.to_csv("image_pseudo_5cv_results.csv", index=False)

    return df_out, result_df
