# Fencing Action Classification Dataset and Model Development

## 1. Introduction

This project investigates the development of a structured dataset and baseline action classification model for Modern Olympic Sabre fencing. The goal is to produce a reproducible pipeline for acquiring bout footage, annotating fencing actions, extracting pose-based representations, and training a classification model inspired by the *FenceNet* framework (Zhou et al., 2022). While comprehensive referee automation requires additional modules such as object tracking and ruleset reasoning, this work focuses strictly on dataset creation and core action recognition.

## 2. Problem Statement

Unlike team sports with well‑established computer vision benchmarks, fencing lacks high‑quality, publicly available datasets containing labeled tactical actions. Sabre in particular presents unique challenges: extremely fast blade motions, complex temporal dependencies, and subjective tactical interpretations governed by right‑of‑way rules. This project aims to address the dataset gap by constructing a curated, labeled collection of sabre exchanges and establishing a baseline model for pose‑based action classification.

## 3. Methodology

The project methodology consists of four major components: video acquisition, annotation, data preprocessing, and pose‑based classification.

### 3.1 Video Collection

High‑quality sabre competition footage is gathered from publicly available broadcasts, including international competitions, collegiate meets, and training sessions with stable camera angles. To ensure utility for computer vision analysis, selected videos prioritize:

* Consistent lateral viewpoint
* Minimal occlusions
* Sufficient resolution for reliable pose estimation
* Clear visibility of both fencers throughout actions

All footage is segmented into short clips representing individual exchanges or tactical sequences.

### 3.2 Action Annotation

Annotations are performed using the CVAT.ai platform. Each clip is labeled according to a standardized, domain‑appropriate action taxonomy that captures essential offensive and defensive events without over‑specifying technical variations.

### 3.3 Data Preprocessing and Pose Extraction

Each annotated clip undergoes a preprocessing pipeline consisting of:

1. **Frame extraction** at a fixed rate suitable for pose estimation.
2. **Human pose estimation** using YOLO‑based keypoint detection models to obtain 2D joint coordinates for both fencers.
3. **Keypoint cleaning** to remove bad detections
4. **Linear interpolation** to fill in missing frames
5. **Data augmentation** to expand the dataset and introduce more variance

The final dataset consists of synchronized pairs of:

* Raw video clips
* Action labels with temporal boundaries
* Pose keypoints for each fencer

This representation enables training of temporal neural networks without dependence on raw pixel data.

### 3.4 Classification Model

Action classification is performed using a pose‑based deep learning model inspired by the *FenceNet* architecture. The classifier operates on sequences of 2D joint coordinates and is designed to capture both spatial body configurations and temporal evolution of fencing actions. Core model components include:

* Spatial attention mechanisms to emphasize action‑relevant joints
* A temporal convolutional module for sequence modeling
* An output classification layer matching the defined action taxonomy

The objective is to establish a strong baseline for sabre action recognition using structured pose information.

## 4. Experimental Setup

Experiments evaluate the classifier using cross‑validation over annotated clips. Metrics include:

* Classification accuracy
* Per‑class precision, recall, and F1‑score
* Confusion matrices to analyze common misclassifications

Evaluation focuses on determining which action categories are most reliably identified and how pose‑based representations perform under conditions of rapid movement and potential occlusion.

## 5. Expected Outcomes

This project aims to produce:

1. A reproducible methodology for collecting and annotating high‑quality sabre footage.
2. A structured, pose‑based dataset suitable for downstream fencing research.
3. A baseline action classifier capable of distinguishing core sabre actions.
4. Insights into the strengths and limitations of pose‑only models for high‑speed combat sports.

## 6. References

Athow, R., McBride, N., & Methven, A. (2016). Using computer vision to assist the scoring of modern fencing. Proceedings of the Midlands Information and Computer Science Conference (MICS), 23–29.

Honda, T., Li, S., Nakaoka, S., & Kuriyama, S. (2020). Motion prediction in competitive fencing. In British Machine Vision Conference (BMVC). BMVA Press.

Malawski, M., & Kwolek, B. (2017). Action segmentation and recognition of fencing footwork. In Telecommunications and Signal Processing (TSP), 2017 40th International Conference on (pp. 710–713). IEEE.

Malawski, M., & Kwolek, B. (2018). Segmentation and classification of fencing footwork. International Journal of Applied Mathematics and Computer Science, 28(1), 149–164. https://doi.org/10.2478/amcs-2018-0012

Zhu, Y., Zhou, Z., & Wu, F. (2022). FenceNet: Fine-grained footwork recognition in fencing. IEEE Transactions on Multimedia, 24, 2561–2572.
