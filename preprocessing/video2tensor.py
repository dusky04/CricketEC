from pathlib import Path
from decord import VideoReader
import argparse
import torch


def uniformly_sample_frames_and_save_as_tensor(
    src: Path, dst: Path, num_frames: int
) -> int:
    i = 0
    all_video_paths = src.rglob("*.avi")
    for video_path in all_video_paths:
        relative_path = video_path.relative_to(src)
        tensor_path = (dst / relative_path).parent / f"{video_path.stem}.pt"
        tensor_path.parent.mkdir(parents=True, exist_ok=True)

        vr = VideoReader(str(video_path))
        indices = torch.linspace(0, len(vr) - 1, num_frames, dtype=torch.int).tolist()
        frames = torch.from_numpy(vr.get_batch(indices=indices).asnumpy()).permute(
            0, 3, 1, 2
        )
        torch.save(frames, tensor_path)
        print(f"Processing: {video_path} -> {tensor_path}")
        i += 1
    return i


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility which loads indices from the videos provided using uniform sampling and makes tensors out of those frames and saves them"
    )

    parser.add_argument(
        "--src",
        type=Path,
        default="CricketEC",
        help="Path to the source directory containing videos",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default="UniformSamplingTensorCricketEC",
        help="Path to the destination directory for output",
    )
    parser.add_argument(
        "--frames",
        "-n",
        type=int,
        default=16,
        help="Number of frames to extract per video",
    )

    args = parser.parse_args()

    src: Path = args.src
    dst: Path = args.dst
    num_frames = args.frames

    if not src.exists():
        raise FileNotFoundError(f"Source directory '{src}' does not exist.")

    dst.mkdir(parents=True, exist_ok=True)

    print("-" * 50)
    print(f"Source            : {src}")
    print(f"Destination       : {dst}")
    print(f"Frames to extract : {num_frames}")
    print("-" * 50)

    num_video_processed = uniformly_sample_frames_and_save_as_tensor(
        src, dst, num_frames
    )

    print("-" * 50)
    print(f"Processed {num_video_processed} video(s).")
    print("-" * 50)
