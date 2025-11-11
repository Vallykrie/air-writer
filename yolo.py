from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)
results = model.train(data="hand-keypoints.yaml", epochs=100, imgsz=640)