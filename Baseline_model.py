from imports import *

model = YOLO("yolov8s.pt")
model.train(data="dataset.yaml",epochs=50,
    batch=5,
    imgsz=512,
    device="cuda" if torch.cuda.is_available() else "cpu",
    name="weed_detection")

results = model.val(data="dataset.yaml")
f1_score = results.box.f1.mean()
map50_95 = results.box.map
total_score = f1_score*0.5 + map50_95*0.5
print(f"F1-score: {f1_score:.4f}")
print(f"mAP@[.5:.95]: {map50_95:.4f}")
print(f"Total-score: {total_score:.4f}")