# =========================================================
# main.py
# =========================================================

import os
import random
import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import StratifiedKFold

from config import CFG
from dataset import build_dataloader
from model import build_model
from train import train_one_fold
from evaluate import evaluate_oof


# =========================================================
# seed
# =========================================================

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# main
# =========================================================

def main():

    cfg = CFG()

    seed_everything(cfg.seed)

    print("=" * 60)
    print(cfg.project_name)
    print("=" * 60)

    # =====================================================
    # load metadata
    # =====================================================

    df = pd.read_csv(cfg.meta_csv)

    print("df shape:", df.shape)

    # binary label
    if cfg.target_col not in df.columns:

        df[cfg.target_col] = (
            df[cfg.bacterial_ratio_col]
            >= cfg.threshold_bacterial
        ).astype(int)

    print(df[cfg.target_col].value_counts())

    # =====================================================
    # split
    # =====================================================

    skf = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.seed,
    )

    oof_probs = np.zeros(len(df))

    # =====================================================
    # fold loop
    # =====================================================

    for fold, (train_idx, valid_idx) in enumerate(

        skf.split(df, df[cfg.target_col])

    ):

        print("\n")
        print("=" * 60)
        print(f"FOLD {fold + 1}")
        print("=" * 60)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)

        print("train:", len(train_df))
        print("valid:", len(valid_df))

        # =================================================
        # pseudo label
        # =================================================

        if cfg.use_pseudo:

            pseudo_df = pd.read_csv(cfg.pseudo_csv)

            print("pseudo:", len(pseudo_df))

            train_df = pd.concat(
                [train_df, pseudo_df],
                axis=0
            ).reset_index(drop=True)

            print("train + pseudo:", len(train_df))

        # =================================================
        # dataloader
        # =================================================

        train_loader = build_dataloader(
            cfg=cfg,
            df=train_df,
            mode="train"
        )

        valid_loader = build_dataloader(
            cfg=cfg,
            df=valid_df,
            mode="valid"
        )

        # =================================================
        # model
        # =================================================

        model = build_model(cfg)

        # =================================================
        # training
        # =================================================

        valid_prob = train_one_fold(
            cfg=cfg,
            fold=fold,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
        )

        oof_probs[valid_idx] = valid_prob

    # =====================================================
    # evaluation
    # =====================================================

    evaluate_oof(
        y_true=df[cfg.target_col].values,
        y_prob=oof_probs,
        save_path=cfg.result_dir / "oof_metrics.csv"
    )

    print("\nDONE")


# =========================================================
# run
# =========================================================

if __name__ == "__main__":
    main()
