import json
import os
from collections import Counter
from ultralytics import YOLO

VIDEO_PATH = "data/kitchen_data.mp4"
MODEL_PATH = "versions/yolov8n.pt"
OUTPUT_PATH = "output/report.json"
FRAME_SKIP = 5

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


yolo_model = YOLO(MODEL_PATH)

def classify_environment(objects):
    home_objs = {"bed", "sofa", "tv", "refrigerator", "microwave", "oven", "dining table", "chimney"}
    shop_objs = {"shelf", "bottle", "box"}
    office_objs = {"laptop", "keyboard", "chair", "desk"}
    
    if home_objs.intersection(objects):
        return "Home"
    elif shop_objs.intersection(objects):
        return "Shop"
    elif office_objs.intersection(objects):
        return "Office"
    else:
        return "Unknown"

print("Tracking objects across video...")
results = yolo_model.track(source=VIDEO_PATH, show=False, persist=True)

unique_object_ids = {}

for r in results:
    names = r.names
    for box in r.boxes:
        obj_name = names[int(box.cls[0])]
        obj_id = int(box.id[0])
        if obj_name not in unique_object_ids:
            unique_object_ids[obj_name] = set()
        unique_object_ids[obj_name].add(obj_id)

object_counts = {obj: len(ids) for obj, ids in unique_object_ids.items()}

environment = classify_environment(set(object_counts.keys()))

report = {
    "environment": environment,
    "items": object_counts
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(report, f, indent=4)

print(f"Report saved to {OUTPUT_PATH}")
print(json.dumps(report, indent=4))