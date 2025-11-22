import pickle
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import torch
from decord import VideoReader
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import get_classes


class FrameSampling(Enum):
    UNIFORM = 0
    JITTERED = 1
    PIXEL_INTENSITY = 2


class CricketEC(Dataset):
    def __init__(
        self,
        c,
        dir: Path,
        transform: Optional[transforms.Compose] = None,
        sampling: FrameSampling = FrameSampling.UNIFORM,
    ) -> None:
        self.dataset_dir = dir
        # paths of all the videos present in the dataset - label / video.avi
        self.video_paths = list(dir.glob("*/*.avi"))
        self.class_names, self.class_to_idx = get_classes(dir)
        self.transform = transform
        self.sampling = sampling
        self.config = c

        self.index_map = None
        cache_path = Path(
            f"CricketEC/pixel_intensity_indices_{self.config.NUM_FRAMES}_frames.pkl"
        )
        if self.sampling == FrameSampling.PIXEL_INTENSITY:
            print("Loaded indices:", str(cache_path))
            with open(cache_path, "rb") as f:
                self.index_map = pickle.load(f)

    def __len__(self) -> int:
        return len(self.video_paths)

    def load_video_frames(self, idx: int) -> torch.Tensor:
        vr = VideoReader(str(self.video_paths[idx]))
        video_path = self.video_paths[idx]

        indices = []
        match self.sampling:
            case FrameSampling.UNIFORM:
                indices = torch.linspace(
                    0, len(vr) - 1, self.config.NUM_FRAMES, dtype=torch.float32
                ).tolist()
            case FrameSampling.JITTERED:
                raise NotImplementedError
            case FrameSampling.PIXEL_INTENSITY:
                if self.index_map is None:
                    raise FileNotFoundError("INDEX MAP NOT LOADING")

                key = str(video_path.relative_to(self.dataset_dir.parent)).replace(
                    "\\", "/"
                )

                if key not in self.index_map:
                    raise KeyError(f"Video {key} not found in pre-computed index map.")

                indices = self.index_map[key]

                if len(indices) != self.config.NUM_FRAMES:
                    if len(indices) > self.config.NUM_FRAMES:
                        indices = indices[: self.config.NUM_FRAMES]
                    elif len(indices) < self.config.NUM_FRAMES:
                        indices += [indices[-1]] * (
                            self.config.NUM_FRAMES - len(indices)
                        )

        frames = torch.from_numpy(vr.get_batch(indices=indices).asnumpy()).permute(
            0, 3, 1, 2
        )
        if self.transform:
            return self.transform(frames)
        return frames

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        if self.config.NUM_FRAMES > 32:
            raise Exception("CANT HANDLE 32 FRAMES")

        video = self.load_video_frames(index)
        label = self.video_paths[index].parent.name
        label_idx = self.class_to_idx[label]
        return video, label_idx


class KeypointsDataset(Dataset):
    def __init__(
        self,
        c,
        dir: Path,
        transform: Optional[transforms.Compose] = None,
        # sampling: FrameSampling = FrameSampling.UNIFORM,
    ) -> None:
        self.dataset_dir = dir
        # paths of all the videos present in the dataset - label / video.avi
        self.feature_paths = list(dir.rglob("*.pt"))
        self.class_names, self.class_to_idx = get_classes(dir)
        self.transform = transform
        # self.sampling = sampling
        self.config = c

    def __len__(self) -> int:
        return len(self.feature_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # 1. Load the tensor (ensure CPU to avoid DataLoader worker crashes)
        # Expected Shape: (num_frames, num_keypoints, 2)
        keypoints = torch.load(self.feature_paths[index], map_location="cpu")

        num_frames = keypoints.shape[0]
        target_frames = self.config.NUM_FRAMES

        # 2. Frame Sampling / Padding Logic
        if num_frames > target_frames:
            # DOWNSAMPLE: Uniformly select 'target_frames' from available frames
            indices = torch.linspace(0, num_frames - 1, target_frames).long()
            keypoints = keypoints[indices]  # Indexing along dim 0 automatically

        elif num_frames < target_frames:
            # PADDING: Repeat the last frame to fill the gap
            diff = target_frames - num_frames

            # Get the last frame -> Shape (1, 13, 2)
            last_frame = keypoints[-1].unsqueeze(0)

            # Repeat it 'diff' times -> Shape (diff, 13, 2)
            padding = last_frame.repeat(diff, 1, 1)

            # Concatenate -> Shape (target_frames, 13, 2)
            keypoints = torch.cat([keypoints, padding], dim=0)

        # (If num_frames == target_frames, we do nothing)

        # 3. Get Label
        label = self.feature_paths[index].parent.name
        label_idx = self.class_to_idx[label]

        # 4. Apply Augmentations/Transforms
        if self.transform:
            keypoints = self.transform(keypoints)

        return keypoints, label_idx


def get_dataloaders(
    config,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
    sampling: FrameSampling = FrameSampling.UNIFORM,
) -> Tuple[DataLoader[CricketEC], DataLoader[CricketEC]]:
    train_dir = Path(config.DATASET_NAME) / "train"
    test_dir = Path(config.DATASET_NAME) / "test"

    train_dataset = CricketEC(
        c=config, dir=train_dir, transform=train_transform, sampling=sampling
    )
    test_dataset = CricketEC(
        c=config, dir=test_dir, transform=test_transform, sampling=sampling
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    return train_dataloader, test_dataloader


def get_val_dataloader(
    config,
    val_transform: transforms.Compose,
    sampling: FrameSampling = FrameSampling.UNIFORM,
) -> DataLoader[CricketEC]:
    val_dir = Path(config.DATASET_NAME) / "val"
    val_dataset = CricketEC(
        c=config, dir=val_dir, transform=val_transform, sampling=sampling
    )
    return DataLoader(
        val_dataset,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )


def get_keypoint_dataloaders(
    config,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
) -> Tuple[DataLoader[KeypointsDataset], DataLoader[KeypointsDataset]]:
    train_dir = Path(config.DATASET_NAME) / "train"
    test_dir = Path(config.DATASET_NAME) / "test"

    train_dataset = KeypointsDataset(
        c=config,
        dir=train_dir,
        transform=train_transform,
    )
    test_dataset = KeypointsDataset(c=config, dir=test_dir, transform=test_transform)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    return train_dataloader, test_dataloader
