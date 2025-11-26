from dataclasses import dataclass

import torch
from torch import nn
from pathlib import Path
from utils import setup_and_download_dataset
from dataset import FrameSampling, get_dataloaders
from models.video_resnet import VideoResNetMyModel
from torchvision.transforms import v2
from train import train_model


@dataclass
class C:
    LR = 1e-3
    DATASET_NAME = "CricketEC"
    NUM_CLASSES = 15
    NUM_FRAMES = 16
    BATCH_SIZE = 12
    LSTM_HIDDEN_DIM = 128
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.3
    FC_DROPOUT = 0.4
    TRAIN_SIZE = 0.8
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 6
    NUM_EPOCHS = 20
    WEIGHT_DECAY = 1e-4


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup the dataset
    DATASET_NAME = "CricketEC"
    CRICKET_EC_URL = "https://drive.google.com/file/d/1QRM360a5HvRKvPF3k7vOT1PS0suxioJd/view?usp=sharing"

    setup_and_download_dataset(
        DATASET_NAME, url=CRICKET_EC_URL, download_dir=Path("zipped_data")
    )

    c = C()

    # setup transforms
    train_transform = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0), antialias=True),
            v2.RandomRotation(degrees=15),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(0.2, 0.2, 0.2, 0.1),
            v2.RandomGrayscale(p=0.2),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        ]
    )
    test_transform = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # setup dataloaders
    train_dataloader, test_dataloader = get_dataloaders(
        c,
        train_transform=train_transform,
        test_transform=test_transform,
        sampling=FrameSampling.UNIFORM,
    )

    # setup model
    model = VideoResNetMyModel(c).to(device)

    # loss function
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # optimizer
    # Define parameter groups with different learning rates
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=c.LR,
        weight_decay=c.WEIGHT_DECAY,
    )

    # lr-scheduler

    # train
    train_model(
        c=c,
        exp_name="video_resnet-16frames",
        weights_dir=Path("weights"),
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=None,
        device=device,
    )
