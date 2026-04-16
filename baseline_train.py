# baseline_train.py
# Phase 1: Baseline YOLOv8n training on Hard Hat Workers dataset

import os
from roboflow import Roboflow
from ultralytics import YOLO

# ── 1. Download dataset from Roboflow ──────────────────────────────────────────
rf = Roboflow(api_key="HwfWq8KGnQMCnz3mwrWd")
project = rf.workspace("joseph-nelson").project("hard-hat-workers")
dataset = project.version(1).download("yolov8")

data_yaml = os.path.abspath(f"{dataset.location}/data.yaml")

# Patch data.yaml to use absolute paths so YOLO can find images regardless of cwd
dataset_dir = os.path.abspath(dataset.location)
with open(data_yaml, "r") as f:
    yaml_content = f.read()

yaml_content = yaml_content.replace("../train/images", f"{dataset_dir}/train/images")
yaml_content = yaml_content.replace("../test/images",  f"{dataset_dir}/test/images")
yaml_content = yaml_content.replace("../valid/images", f"{dataset_dir}/test/images")

with open(data_yaml, "w") as f:
    f.write(yaml_content)

# ── 2. Train baseline YOLOv8n (all defaults) ───────────────────────────────────
model = YOLO("yolov8n.pt")  # pretrained nano model

results = model.train(
    data=data_yaml,
    epochs=20,
    imgsz=416,
    batch=32,
    device="mps",           # Apple Silicon M3
    project="runs/baseline",
    name="yolov8n_hardhat",
    exist_ok=True,
)

# ── 3. Evaluate on validation set ─────────────────────────────────────────────
metrics = model.val(
    data=data_yaml,
    device="mps",
)

print("\n" + "="*50)
print("BASELINE EVALUATION RESULTS")
print("="*50)
print(f"mAP@0.5       : {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95  : {metrics.box.map:.4f}")
print("="*50)
print(f"\nFull results saved to: runs/baseline/yolov8n_hardhat/")
