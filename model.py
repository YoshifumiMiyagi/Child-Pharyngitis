# =========================================================
# model.py
# =========================================================

import timm
import torch
import torch.nn as nn


# =========================================================
# fusion model
# =========================================================

class FusionModel(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            num_classes=0,
        )

        feat_dim = self.backbone.num_features

        self.table_net = nn.Sequential(
            nn.Linear(len(cfg.table_cols), 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.head = nn.Sequential(
            nn.Linear(feat_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x_img, x_tab):

        feat_img = self.backbone(x_img)

        feat_tab = self.table_net(x_tab)

        feat = torch.cat(
            [feat_img, feat_tab],
            dim=1
        )

        out = self.head(feat)

        return out.squeeze(1)


# =========================================================
# image model
# =========================================================

class ImageModel(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            num_classes=1,
        )

    def forward(self, x):

        x = self.backbone(x)

        return x.squeeze(1)


# =========================================================
# build model
# =========================================================

def build_model(cfg):

    if cfg.use_table:

        return FusionModel(cfg)

    return ImageModel(cfg)
