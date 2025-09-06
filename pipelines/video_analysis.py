import cv2
from ultralytics import YOLO
from collections import Counter
import json
import os

# -------------------------
# Config
# -------------------------
VIDEO_PATH = "data/input_video.mp4"
MODEL_PATH = YOLO("versions/yolov8n.pt")  # Pretrained YOLOv8n model
OUTPUT_PATH = "output/report.json"
FRAME_RATE = 2  # frames per second to process

# Make sure output folder exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# -------------------------
# Load YOLOv8 Model
# -------------------------
yolo_model = YOLO(MODEL_PATH)

# -------------------------
# Environment Classification by Object Mapping
# -------------------------
def classify_environment(objects):
    if any(obj in objects for obj in ["bed", "sofa", "tv", "refrigerator", "microwave"]):
        return "Home"
    elif any(obj in objects for obj in ["shelf", "bottle", "box", "refrigerator"]):
        return "Shop"
    elif any(obj in objects for obj in ["laptop", "keyboard", "chair", "desk"]):
        return "Office"
    else:
        return "Unknown"

# -------------------------
# Video Processing
# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
interval = max(int(fps / FRAME_RATE), 1)

all_objects = []
frame_count = 0

print("Processing video frames...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames according to interval
    if frame_count % interval == 0:
        results = yolo_model(frame, verbose=False)
        
        for r in results:
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                obj_name = names[cls_id]
                all_objects.append(obj_name)

    frame_count += 1

cap.release()
print(f"Processed {frame_count} frames, detected {len(all_objects)} objects.")

# -------------------------
# Aggregate and Classify
# -------------------------
object_counts = Counter(all_objects)
environment = classify_environment(object_counts.keys())

# -------------------------
# Build JSON Report
# -------------------------
report = {
    "environment": environment,
    "items": dict(object_counts)
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(report, f, indent=4)

print(f"Report saved to {OUTPUT_PATH}")
print(json.dumps(report, indent=4))

import os

# Set the output file path
OUTPUT_PATH = "output/report.json"

# Make sure the folder exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Later in the script, save JSON
import json
with open(OUTPUT_PATH, "w") as f:
    json.dump(report, f, indent=4)

print(f"Report saved to {OUTPUT_PATH}")