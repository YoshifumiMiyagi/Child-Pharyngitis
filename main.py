import argparse
import pandas as pd
import os
from config import CFG

from run_image_cv import run_image_cv
from run_fusion_cv import run_fusion_cv
from run_image_pseudo_cv import run_image_pseudo_cv


# =====================================================
# argparse
# =====================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="image",
        choices=["image", "fusion", "pseudo"],
    )
    
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/kaggle/input/pharyngitis-images-metadata/PGUPharyngitis",
    )
    
    parser.add_argument(
        "--excel_path",
        type=str,
        default="/kaggle/input/pharyngitis-images-metadata/PGUPharyngitis/excel.xlsx",
    )
    
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/kaggle/input/pharyngitis-images-metadata/PGUPharyngitis/data_image_pharyngitis_nature",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    return parser.parse_args()
# =====================================================
# main
# =====================================================

def main():

    args = parse_args()

    cfg = CFG()

    cfg.seed = args.seed
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.n_splits = args.n_splits
    cfg.image_dir = args.image_dir
    cfg.dataset_dir = args.dataset_dir
    # =================================================
    # load data
    # =================================================

    df = pd.read_excel(args.excel_path)
    
    diagnosis_cols = [c for c in df.columns if "Diagnosis" in c]
    diag = df[diagnosis_cols].replace({
        "Bacterial": 1,
        "Non Bacterial": 0,
      })
    
    diag = diag.apply(pd.to_numeric, errors="coerce")
    
    df["bacterial_ratio"] = diag.mean(axis=1)
    df["label"] = (df["bacterial_ratio"] >= 0.5).astype(int)
    if "bacterial_ratio" not in df.columns:
        df["bacterial_ratio"] = df[diagnosis_cols].mean(axis=1)
    df["label"] = (
        df["bacterial_ratio"] >= 0.5
    ).astype(int)

    df["stratify_label"] = df["label"]
    possible_id_cols = [
        "sample_id",
        "SampleID",
        "sample",
        "patient_ID",
        "id",
        "image_id",
    ]
    
    id_col = None
    
    for c in possible_id_cols:
        if c in df.columns:
            id_col = c
            break
    
    print("ID COLUMN:", id_col)
    

    def find_image_path(image_dir, pid):
        pid = str(int(pid))
        folder = os.path.join(image_dir, pid)
    
        patterns = [
            os.path.join(folder, "*.jpg"),
            os.path.join(folder, "*.JPG"),
            os.path.join(folder, "*.jpeg"),
            os.path.join(folder, "*.JPEG"),
            os.path.join(folder, "*.png"),
            os.path.join(folder, "*.PNG"),
        ]
    
        files = []
        for p in patterns:
            files.extend(glob.glob(p))
    
        if len(files) == 0:
            return None
    
        return files[0]
    
    
    df["image_path"] = df[id_col].apply(
        lambda x: find_image_path(args.image_dir, x)
    )
    
    print("missing images:", df["image_path"].isna().sum())
    print(df[df["image_path"].isna()][id_col].head(20))
    
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)
    
    # =================================================
    # transform
    # =================================================

    train_tfms = cfg.get_train_transforms()
    valid_tfms = cfg.get_valid_transforms()

    # =================================================
    # run
    # =================================================

    if args.mode == "image":

        run_image_cv(
            df=df,
            train_tfms=train_tfms,
            valid_tfms=valid_tfms,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            lr=cfg.lr,
            n_splits=cfg.n_splits,
        )

    elif args.mode == "fusion":

        run_fusion_cv(
            df=df,
            tabular_cols=cfg.tabular_cols,
            num_cols=cfg.num_cols,
            train_tfms=train_tfms,
            valid_tfms=valid_tfms,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            lr=cfg.lr,
            n_splits=cfg.n_splits,
        )

    elif args.mode == "pseudo":

        real_df = (
            df[df["Age"] <= 18]
            .reset_index(drop=True)
        )

        adult_df = (
            df[df["Age"] > 18]
            .reset_index(drop=True)
        )

        run_image_pseudo_cv(
            real_df=real_df,
            adult_df=adult_df,
            train_tfms=train_tfms,
            valid_tfms=valid_tfms,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            lr=cfg.lr,
            n_splits=cfg.n_splits,
        )


# =====================================================
# run
# =====================================================

if __name__ == "__main__":
    main()
