from enum import Enum
from typing import cast

import torch
from torch import nn
from torchvision.models import EfficientNet, ResNet, efficientnet_b0, resnet34


class BackboneType(Enum):
    RESNET_34 = 0
    EFFNET_B0 = 1


class MotionType(Enum):
    LSTM = 0
    GRU  = 0


class CricketECModel(nn.Module):
    def __init__(self, C, backbone_type: BackboneType, motion_type: MotionType) -> None:
        super().__init__()

        self.in_features: int
        backbone: ResNet | EfficientNet | None = None
        match backbone_type:
            case BackboneType.RESNET_34:
                backbone = load_resnet34()
                self.in_features = backbone.fc.in_features
            case BackboneType.EFFNET_B0:
                backbone = load_effnet()
                self.in_features = cast(nn.Linear, backbone.classifier[1]).in_features

        if backbone:
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        self.batch_norm = nn.BatchNorm2d(self.in_features)

        self.motion_classifier: nn.LSTM | nn.GRU
        match motion_type:
            case MotionType.LSTM:
                self.motion_classifier = nn.LSTM(
                    input_size=self.in_features,
                    hidden_size=C.LSTM_HIDDEN_DIM,
                    num_layers=C.LSTM_NUM_LAYERS,
                    dropout=C.LSTM_DROPOUT if C.LSTM_NUM_LAYERS > 1 else 0.0,
                    batch_first=True,
                    bidirectional=True,
                )
            case MotionType.GRU:
                self.motion_classifier = nn.GRU(
                    input_size=self.in_features,
                    hidden_size=C.LSTM_HIDDEN_DIM,
                    num_layers=C.LSTM_NUM_LAYERS,
                    dropout=C.LSTM_DROPOUT if C.LSTM_NUM_LAYERS > 1 else 0.0,
                    batch_first=True,
                    bidirectional=True,
                )

        self.temporal_out_dim = C.LSTM_HIDDEN_DIM * 2

        self.layer_norm = nn.LayerNorm(self.temporal_out_dim)

        self.attention_pool = nn.Sequential(
            nn.Linear(self.temporal_out_dim, 1), nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(C.FC_DROPOUT)

        self.linear = nn.Sequential(
            nn.Linear(self.temporal_out_dim, self.temporal_out_dim // 2),
            nn.BatchNorm1d(self.temporal_out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(C.FC_DROPOUT),
            nn.Linear(self.temporal_out_dim // 2, C.NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input dim: [batch_size, frame, C, H, W]
        B, T, C, H, W = x.shape

        # since we are passing through conv2D layers in ResNet
        # have to convert dims to [batch_size * frame, C, H, W]
        # output dims: [batch_size, 512 (in_features), 1, 1]
        features = self.feature_extractor(x.view(B * T, C, H, W))
        features = self.batch_norm(features)

        # LSTM expects (batch_size, sequence_length, input_size) as input
        features = features.view(B, T, -1)  # [batch_size, sequence_length, 512]
        # output of LSTM dims: [batch_size, sequence_length, hidden_dim]
        features, _ = self.motion_classifier(features)
        features = self.layer_norm(features)

        attention_weights = self.attention_pool(features)
        features = torch.sum(features * attention_weights, dim=1)
        # features = torch.mean(features, dim=1)  # pool across time

        features = self.dropout(features)
        output = self.linear(features)
        return output


def load_resnet34() -> ResNet:
    resnet = resnet34(weights="DEFAULT")
    for param in resnet.parameters():
        param.requires_grad = False
    for param in resnet.layer4.parameters():
        param.requires_grad = True
    for param in resnet.layer3.parameters():
        param.requires_grad = True
    return resnet


def load_effnet() -> EfficientNet:
    effnet = efficientnet_b0(weights="DEFAULT")
    for param in effnet.parameters():
        param.requires_grad = False
    for idx in range(6, 9):
        for param in effnet.features[idx].parameters():
            param.requires_grad = True
    return effnet
