# config.py

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CFG:
    # =====================
    # Project
    # =====================
    project_name: str = "pediatric_pharyngitis_ai"
    seed: int = 42
    n_splits: int = 5

    # =====================
    # Paths
    # =====================
    root_dir: Path = Path("/kaggle/working")
    data_dir: Path = Path("/kaggle/input/pharyngitis-data")
    image_dir: Path = data_dir / "images"
    meta_csv: Path = data_dir / "metadata.csv"

    output_dir: Path = root_dir / "outputs"
    model_dir: Path = output_dir / "models"
    cam_dir: Path = output_dir / "cams"
    result_dir: Path = output_dir / "results"

    # =====================
    # Target
    # =====================
    target_col: str = "label"
    bacterial_ratio_col: str = "bacterial_ratio"
    threshold_bacterial: float = 0.5

    # =====================
    # Image
    # =====================
    img_size: int = 224
    in_channels: int = 3

    # =====================
    # Training
    # =====================
    model_name: str = "resnet18"
    pretrained: bool = True
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2

    # =====================
    # Pseudo-label
    # =====================
    use_pseudo: bool = False
    pseudo_csv: Path = data_dir / "pseudo_labels.csv"
    pseudo_weight: float = 0.5
    pseudo_pos_threshold: float = 0.9
    pseudo_neg_threshold: float = 0.1

    # =====================
    # Fusion / Table
    # =====================
    use_table: bool = False
    table_cols: tuple = (
       'Age','Gender', 'Feeling Cold', 'Sore Throat',
       'Cephalalgia', 'Myalgia', 'Rhinorrhea', 'Diarrhea', 'Vomit', 'Cough',
       'Pyrexia', 'Rigors', 'Lethargy', 'Sputum', 'Abdominal Pain', 'Vertigo',
       'Sternutation', 'Eye Infection', 'Otalgia', 'Nasal Congestion',
       'Asthma', 'Anorexia',
    )

    tabular_cols: tuple = (
       'Age','Gender', 'Feeling Cold', 'Sore Throat',
       'Cephalalgia', 'Myalgia', 'Rhinorrhea', 'Diarrhea', 'Vomit', 'Cough',
       'Pyrexia', 'Rigors', 'Lethargy', 'Sputum', 'Abdominal Pain', 'Vertigo',
       'Sternutation', 'Eye Infection', 'Otalgia', 'Nasal Congestion',
       'Asthma', 'Anorexia',
    )

    num_cols: tuple = (
        "Age",
    )
    
    # =====================
    # Grad-CAM / ROI
    # =====================
    use_gradcam: bool = True
    cam_method: str = "gradcam++"
    roi_mask_dir: Path = data_dir / "roi_masks"

    roi_labels: dict = None

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cam_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        if self.roi_labels is None:
            self.roi_labels = {
                "soft_palate": 60,
                "posterior_wall": 240,
                "uvula": 120,
                "tongue": 180,
            }


    def get_train_transforms(self):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])

    def get_valid_transforms(self):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(),
            ToTensorV2(),
        ])
