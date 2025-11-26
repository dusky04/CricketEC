import torch
from torch import nn
from torchvision.models.video import r3d_18, VideoResNet


class VideoResNetMyModel(nn.Module):
    def __init__(self, C) -> None:
        super().__init__()

        video_resnet = r3d_18(weights="DEFAULT")
        for param in video_resnet.parameters():
            param.requires_grad = False
        for param in video_resnet.layer4.parameters():
            param.requires_grad = True

        self.out_dim = 512

        video_resnet.fc = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim // 2),
            nn.BatchNorm1d(self.out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(C.FC_DROPOUT),
            nn.Linear(self.out_dim // 2, C.NUM_CLASSES),
        )

        self.model = video_resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input dim: [batch_size, frame, C, H, W]
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)


def load_video_resnet(c) -> VideoResNet:
    video_resnet = r3d_18(weights="DEFAULT")
    for param in video_resnet.parameters():
        param.requires_grad = False
    for param in video_resnet.layer4.parameters():
        param.requires_grad = True
    video_resnet.fc = nn.Linear(in_features=512, out_features=c.NUM_CLASSES)
    return video_resnet
