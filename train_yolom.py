from ultralytics import YOLO
# Config
DATA_YAML = "/content/Suture_Analysis/data/data.yaml"
MODEL     = "yolo11m.pt"                                
IMGSZ     = 640
EPOCHS    = 120
BATCH     = 16
DEVICE    = 0
RUN_NAME  = "suture_yolo11s_rf_aug"
PROJECT   = "runs_detect"
# Model Training
model = YOLO(MODEL)
results = model.train(
    data=DATA_YAML,
    imgsz=IMGSZ,
    epochs=EPOCHS,
    batch=BATCH,
    device=DEVICE,
    project=PROJECT,
    name=RUN_NAME,
    patience=50,
    # light augments
    degrees=5, translate=0.06, scale=0.20, shear=0.05,
    fliplr=0.30, flipud=0.0,
    hsv_h=0.01, hsv_s=0.20, hsv_v=0.20,
    mosaic=0.20, mixup=0.0, close_mosaic=10,
    # stable optimizer/schedule
    lr0=0.003, lrf=0.12, momentum=0.937, weight_decay=0.0005, cos_lr=True,
)


# Validation
metrics = model.val(data=DATA_YAML, imgsz=IMGSZ, device=DEVICE)
print("\n--- Validation ---")
print(f"mAP50    : {float(metrics.box.map50):.3f}")
print(f"mAP50-95 : {float(metrics.box.map):.3f}")
print(f"Precision: {float(metrics.box.mp):.3f}")
print(f"Recall   : {float(metrics.box.mr):.3f}")
