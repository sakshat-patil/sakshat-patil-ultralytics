# SP26 Homework 2 — Object Detection
**Course:** Deep Learning, San José State University, Spring 2026  
**Due:** April 17, 2026  

---

## Objective

Explore optimization techniques for 2D object detection.

## Selected Option: Option 1 — Training Optimization

Improve the training process of a 2D object detection model through:

- Modifying the model architecture via configurations (e.g., backbone, neck, head, feature fusion)
- Changing the training strategy (e.g., data augmentation, optimizer, scheduler, loss settings, batch size, image size, transfer learning strategy)

---

## Submission Contents

| File | Description |
|---|---|
| `baseline_train.py` | Baseline YOLOv8n training script |
| `report.md` | Full optimization report with comparison of COCO-style evaluation results |

---

## Summary

- **Dataset:** Hard Hat Workers (Roboflow Universe)
- **Baseline:** YOLOv8n, 10 epochs, default settings → mAP@0.5: 0.6406, mAP@0.5:0.95: 0.4177
- **Optimized:** YOLOv8s, 20 epochs, AdamW + Cosine Annealing + enhanced augmentation → mAP@0.5: 0.6500, mAP@0.5:0.95: 0.4333

See `report.md` for the full breakdown of what was changed, why, and how it affected performance.
