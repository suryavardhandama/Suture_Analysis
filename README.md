# Suture Analysis — Automated Feedback for Medical Training

This project aims to assist **medical students** in evaluating the **accuracy and quality of sutures** from uploaded images.  
It uses **YOLOv11-based object detection** to identify sutures, incision lines, and scale markers, then calculates **stitch length** and **angle deviation** to assess stitching precision.

---

## Overview

| Task | Description |
|------|--------------|
| **Goal** | Detect sutures and incision lines from training pad images |
| **Model** | YOLOv11 (small `YOLOs` & medium `YOLOm` variants) |
| **Input** | Suture training pad images (annotated with Roboflow) |
| **Output** | CSV with each suture’s length (mm) and angle (°) from the incision |

---

## Model Variations

| Model | Description | Precision | Recall | mAP@50 | mAP@50–95 |
|:--|:--|:--:|:--:|:--:|:--:|
| **YOLOs (small)** | Faster, lightweight | 0.57 | 0.418 | 0.358 | 0.176 |
| **YOLOm (medium)** | More accurate, better localization | **0.588** | 0.399 | **0.422** | **0.204** |

> **YOLOm performed best overall**, especially for small sutures and the incision line.
