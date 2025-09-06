from ultralytics import YOLO
import cv2
import json
from collections import Counter

# Load YOLOv8 pretrained model (nano version for speed)
model = YOLO("yolov8n.pt")

# Define simple mapping for environment classification
def classify_environment(objects):
    if any(obj in objects for obj in ["bed", "sofa", "tv", "refrigerator"]):
        return "Home"
    elif any(obj in objects for obj in ["shelf", "bottle", "box"]):
        return "Shop"
    elif any(obj in objects for obj in ["laptop", "keyboard", "chair", "desk"]):
        return "Office"
    else:
        return "Unknown"

# Open video
cap = cv2.VideoCapture("input_video.mp4")

all_objects = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = model(frame, verbose=False)
    
    for r in results:
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            obj_name = names[cls_id]
            all_objects.append(obj_name)

cap.release()

# Count detected items
object_counts = Counter(all_objects)

# Decide environment
environment = classify_environment(object_counts.keys())

# Build report
report = {
    "environment": environment,
    "items": dict(object_counts)
}

# Save JSON
with open("report.json", "w") as f:
    json.dump(report, f, indent=4)

print(json.dumps(report, indent=4))
