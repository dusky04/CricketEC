import torch
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report



def uniformly_sample_keypoints(num_frames: int = 16) -> tuple[np.ndarray, np.ndarray]:
    src = Path("Keypoints")
    X = []
    y = []
    for keypoints_path in src.rglob("*.pt"):
        # shape of keypoints: [detected_frames, 13, 2]
        keypoints: torch.Tensor = torch.load(keypoints_path)

        # uniformly sample points
        if len(keypoints) >= num_frames:
            indices = torch.linspace(0, len(keypoints) - 1, num_frames, dtype=torch.int)
            keypoints = keypoints[indices]
        else:
            num_keypoints_to_pad = num_frames - len(keypoints)
            padding_keypoints = torch.zeros(num_keypoints_to_pad, 13, 2)
            keypoints = torch.cat([keypoints, padding_keypoints])

        # shape: [detected_frames * 13 * 2]
        keypoints = keypoints.flatten().numpy()
        X.append(keypoints)

        if keypoints_path.stem == "hook":
            y.append(0)
        else:
            y.append(1)

    X = np.array(X)
    y = np.array(y)

    return X, y


X, y = uniformly_sample_keypoints(16)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    eval_metric='logloss'
)
