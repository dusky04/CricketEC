# generates the keypoints for pose estimation of
# `PULL` and `HOOK` shots


from ultralytics import YOLO

YOLO_PATH = "yolo/yolo-weights.pt"

yolo = YOLO(YOLO_PATH, verbose=False)


# src = Path.cwd()
# dst = Path("Keypoints")
# dst.mkdir(exist_ok=True)


# hook_pull_videos = src.rglob("*.avi")
# for video in hook_pull_videos:
#     print(video.stem)
#     parent_dir = dst / video.parent.relative_to(src)
#     parent_dir.mkdir(exist_ok=True)
#     # break
#     all_keypoints = []
#     for result in yolo.track(video, stream=True, persist=True, verbose=False):
#         if result.keypoints and len(result.keypoints.xy) > 0:
#             keypoints_xy = result.keypoints.xy
#             person_keypoints = keypoints_xy[0]
#             all_keypoints.append(person_keypoints)

#     processed_tensor = torch.stack(all_keypoints, dim=0)
#     torch.save(processed_tensor, f"{parent_dir / video.stem}.pt")

all_keypoints = []
pose_results = yolo.track(
    "CricketEC/train/hook/hook_0002.avi", stream=True, persist=True, verbose=True
)
for result in pose_results:
    if result.keypoints and len(result.keypoints.xy) > 0:
        keypoints_xy = result.keypoints.xy
        person_keypoints = keypoints_xy[0]

        all_keypoints.append(person_keypoints)
