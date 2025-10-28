import cv2
import math
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

# ======= CONFIG =======
MODEL_PATH = "/content/Suture_Analysis/runs_detect/suture_yolo11m_rf_aug/weights/best.pt"
DATA_PATH  = "/content/Suture_Analysis/data/test/images"  # test images folder
SAVE_PATH  = "/content/Suture_Analysis/suture_measurements.csv"

# ======= HELPER FUNCTIONS =======
def euclidean_distance(p1, p2):
    """Compute Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def find_scale_mm(scale_boxes):
    """Estimate pixels per mm using scale marker (assuming ~10 mm segment visible)"""
    if len(scale_boxes) == 0:
        return 1.0  # fallback
    # take width of first scale box (assuming 10 mm segment)
    x1, y1, x2, y2 = scale_boxes[0]
    pixel_length = euclidean_distance((x1, y1), (x2, y2))
    pixels_per_mm = pixel_length / 10.0
    return pixels_per_mm

def angle_with_incision(suture_box, incision_box):
    """Compute angle between suture and incision line"""
    sx1, sy1, sx2, sy2 = suture_box
    ix1, iy1, ix2, iy2 = incision_box

    suture_vec = np.array([sx2 - sx1, sy2 - sy1])
    incision_vec = np.array([ix2 - ix1, iy2 - iy1])

    dot = np.dot(suture_vec, incision_vec)
    mag = np.linalg.norm(suture_vec) * np.linalg.norm(incision_vec)
    if mag == 0: return np.nan
    cos_theta = np.clip(dot / mag, -1.0, 1.0)
    return round(math.degrees(math.acos(cos_theta)), 2)

# ======= RUN INFERENCE =======
model = YOLO(MODEL_PATH)
image_paths = list(Path(DATA_PATH).glob("*.jpg")) + list(Path(DATA_PATH).glob("*.png"))

results_list = []

for img_path in image_paths:
    res = model.predict(source=str(img_path), conf=0.25, iou=0.5, verbose=False)
    boxes = res[0].boxes.xyxy.cpu().numpy()
    classes = res[0].boxes.cls.cpu().numpy()
    names = model.names

    # collect boxes by class
    incision_boxes = [boxes[i] for i, c in enumerate(classes) if names[int(c)] == "incision_line"]
    scale_boxes    = [boxes[i] for i, c in enumerate(classes) if names[int(c)] == "scale_marker"]
    suture_l_boxes = [boxes[i] for i, c in enumerate(classes) if names[int(c)] == "suture_l"]
    suture_r_boxes = [boxes[i] for i, c in enumerate(classes) if names[int(c)] == "suture_r"]
    suture_d_boxes = [boxes[i] for i, c in enumerate(classes) if names[int(c)] == "suture_d"]

    px_per_mm = find_scale_mm(scale_boxes)
    if not incision_boxes:
        print(f"⚠️ No incision line found in {img_path.name}")
        continue

    incision_box = incision_boxes[0]  # assume single incision line

    for cls_name, box_list in {
        "suture_l": suture_l_boxes,
        "suture_d": suture_d_boxes,
        "suture_r": suture_r_boxes
    }.items():
        for i, box in enumerate(box_list, start=1):
            x1, y1, x2, y2 = box
            length_px = euclidean_distance((x1, y1), (x2, y2))
            length_mm = length_px / px_per_mm
            angle_deg = angle_with_incision(box, incision_box)
            results_list.append({
                "image": img_path.name,
                "suture_type": cls_name,
                "stitch_id": i,
                "length_px": round(length_px, 2),
                "length_mm": round(length_mm, 2),
                "angle_deg": angle_deg
            })

# ======= SAVE RESULTS =======
df = pd.DataFrame(results_list)
df.to_csv(SAVE_PATH, index=False)
print(f"✅ Measurements saved to {SAVE_PATH}")
display(df.head())
