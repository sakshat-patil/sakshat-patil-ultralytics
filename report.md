# SP26 Deep Learning — Homework 2
## Option 1: Training Optimization Report
**Dataset:** Hard Hat Workers (Roboflow Universe, Joseph Nelson)  
**Task:** 2D Object Detection — 3 classes: `head`, `helmet`, `person`  
**Framework:** Ultralytics YOLOv8 (Python)

---

## 1. Baseline Setup

| Parameter | Value |
|---|---|
| Model | YOLOv8n (nano) |
| Pretrained weights | `yolov8n.pt` (COCO) |
| Epochs | 10 |
| Image size | 640 × 640 |
| Batch size | 32 |
| Optimizer | Auto (AdamW, selected by Ultralytics) |
| LR schedule | Default (linear decay) |
| Augmentation | Default (mosaic, HSV, flips) |
| Device | Google Colab T4 GPU |

### Baseline Results

| Metric | Score |
|---|---|
| mAP@0.5 | 0.6406 |
| mAP@0.5:0.95 | 0.4177 |

---

## 2. Optimization Strategy

Three categories of changes were applied: architecture, training strategy, and augmentation.

### 2.1 Architecture Change — YOLOv8n → YOLOv8s

**What changed:** Upgraded from the nano model (`yolov8n.pt`) to the small model (`yolov8s.pt`).

**Why:** YOLOv8s has a wider backbone and neck compared to YOLOv8n — approximately 3× more parameters (11.2M vs 3.2M). This gives the model greater representational capacity to distinguish subtle visual differences between helmeted and un-helmeted workers, particularly at varying scales and orientations. The nano model, while fast, is capacity-limited on a multi-scale detection task with fine-grained class distinctions.

### 2.2 Optimizer — AdamW with Cosine Annealing

**What changed:**
- Optimizer explicitly set to `AdamW` with `lr0=0.001`
- Learning rate schedule changed to cosine annealing (`cos_lr=True`, `lrf=0.01`)

**Why:** AdamW decouples weight decay from the gradient update, which leads to better generalization compared to standard SGD, especially when training with momentum on datasets of moderate size. Cosine annealing gradually reduces the learning rate following a cosine curve rather than linearly, which helps the optimizer settle into a flatter minimum and reduces oscillation near convergence. The combination is well-established in modern object detection training pipelines.

### 2.3 Augmentation Enhancements

**What changed:**

| Parameter | Baseline | Optimized |
|---|---|---|
| `mosaic` | 1.0 | 1.0 (kept) |
| `mixup` | 0.0 | 0.1 |
| `hsv_h` | 0.015 | 0.02 |
| `hsv_s` | 0.7 | 0.8 |
| `hsv_v` | 0.4 | 0.5 |
| `flipud` | 0.0 | 0.1 |
| `label_smoothing` | 0.0 | 0.1 |

**Why:**
- **Mixup (0.1):** Blends pairs of training images and their labels. Encourages the model to produce less overconfident predictions and improves generalization to partially occluded workers.
- **Increased HSV jitter:** Construction site images vary significantly in lighting conditions (outdoor sun, indoor shadow, overcast). Stronger color/brightness augmentation teaches the model to be invariant to these conditions.
- **Vertical flip (0.1):** While rare in practice, occasional vertical flips add minor viewpoint diversity.
- **Label smoothing (0.1):** Softens hard one-hot targets, reducing overconfidence in classification and improving calibration of the confidence scores.

### 2.4 Training Duration

**What changed:** Epochs increased from 10 → 20.

**Why:** YOLOv8s has more parameters than YOLOv8n and requires more gradient updates to converge fully. Running only 10 epochs would leave the larger model undertrained relative to its capacity.

---

## 3. Results Comparison

| Metric | Baseline (YOLOv8n) | Optimized (YOLOv8s) | Change |
|---|---|---|---|
| mAP@0.5 | 0.6406 | 0.6500 | **+0.0094 (+1.5%)** |
| mAP@0.5:0.95 | 0.4177 | 0.4333 | **+0.0156 (+3.7%)** |

---

## 4. Analysis

Both metrics improved after applying the combined optimizations.

**mAP@0.5:0.95 (+3.7%)** improved more than mAP@0.5 (+1.5%), which is the more meaningful gain. mAP@0.5 measures detection quality at a single, relatively lenient IoU threshold. mAP@0.5:0.95 averages across multiple IoU thresholds (0.5 to 0.95 in steps of 0.05), meaning it penalizes imprecise bounding box localization more heavily. The larger relative gain here indicates that the optimized model not only detects objects more reliably, but also localizes them more precisely — likely a direct benefit of the larger YOLOv8s backbone, which extracts richer spatial features.

**Architecture capacity** was the dominant factor. Moving from YOLOv8n to YOLOv8s provides the model with wider feature maps and more convolutional filters in both the backbone (CSPDarknet) and the neck (PAN-FPN), enabling better multi-scale feature fusion. For a dataset with workers at varying distances and scales, this directly benefits detection of small objects (distant workers without helmets).

**Cosine annealing** contributed to more stable final-epoch performance by preventing the learning rate from dropping too aggressively early on and allowing fine-grained weight adjustment in later epochs.

**Augmentation improvements** (mixup, stronger HSV, label smoothing) helped reduce overfitting on the training set, producing a model that generalizes better to the test images, as reflected in the improved validation metrics.

**Limitations:** The gains are moderate. This is expected given the short training duration (10–20 epochs) and the relatively small model family (nano/small). Further gains could be achieved with longer training (50+ epochs), a medium or large YOLOv8 variant, or a custom neck modification for better small-object detection.

---

## 5. Configuration Summary

```python
# Baseline
model = YOLO("yolov8n.pt")
model.train(data=data_yaml, epochs=10, imgsz=640, batch=32, device=0)

# Optimized
model = YOLO("yolov8s.pt")
model.train(
    data=data_yaml, epochs=20, imgsz=640, batch=32, device=0,
    optimizer="AdamW", cos_lr=True, lr0=0.001, lrf=0.01,
    mosaic=1.0, mixup=0.1, hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
    flipud=0.1, label_smoothing=0.1, weight_decay=0.0005,
)
```
