from ultralytics import YOLO

YOLO_PATH = "yolo-weights.pt"

# this is the `yolov8n-pose.pt`
yolo = YOLO(YOLO_PATH)
# yolo.predict
