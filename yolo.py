from ultralytics import YOLO
import torch

print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

model = YOLO("yolo11n-pose.pt")
results = model.train(data="hand-keypoints.yaml", epochs=100, imgsz=640)