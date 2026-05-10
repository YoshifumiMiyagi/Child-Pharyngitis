import torch
import torch.nn as nn
from torchvision import models


class ResNet18Regressor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.sigmoid(x)
        return x.squeeze(1)


class ResNet18Fusion(nn.Module):
    def __init__(self, n_tabular, pretrained=True, tab_hidden=32):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.image_encoder = backbone

        self.tabular_encoder = nn.Sequential(
            nn.Linear(n_tabular, tab_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.head = nn.Sequential(
            nn.Linear(in_features + tab_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, img, tab):
        img_feat = self.image_encoder(img)
        tab_feat = self.tabular_encoder(tab)

        x = torch.cat([img_feat, tab_feat], dim=1)
        x = self.head(x)

        return x.squeeze(1)
