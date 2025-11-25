import torch
from pathlib import Path

def uniformly_sample_keypoints(num_frames: int = 16) -> None:
    src = Path("Keypoints")
    X = []
    for keypoints_path in src.rglob("*.pt"):
        # shape of keypoints: [detected_frames, 13, 2]
        keypoints: torch.Tensor = torch.load(keypoints_path)

        # uniformly sample points
        if len(keypoints) >= num_frames:
            indices = torch.linspace(0, len(keypoints) - 1, num_frames, dtype = torch.int)
            keypoints = keypoints[indices]
        else:
            num_keypoints_to_pad = num_frames - len(keypoints)
            padding_keypoints = torch.zeros(num_keypoints_to_pad, 13, 2)
            keypoints = torch.cat([keypoints, padding_keypoints])

        # shape: [detected_frames * 13 * 2]
        keypoints = keypoints.flatten().numpy()
        X.append(keypoints)

        print(keypoints)

        break


uniformly_sample_keypoints(16)
