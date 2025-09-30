# Automated Refereeing for Olympic Saber

## Project Overview

Modern Olympic saber fencing requires referees to make **split-second decisions** about which fencer has right-of-way when simultaneous touches occur. While electronic scoring systems reliably detect valid touches, they do **not capture the sequence of actions** (attack, parry, riposte, counter-attack) that determine priority. This project aims to explore whether a **computer vision system** can reliably detect fencer movements, classify actions, and assist referees in making right-of-way decisions.

### Goals

* Detect fencer movements and blade trajectories.
* Recognize offensive and defensive fencing actions.
* Apply saber priority rules to produce preliminary point calls.

---

## Background

Existing sports analytics technologies demonstrate the feasibility of automated decision support:

* **Hawk-Eye** in tennis and soccer uses multi-camera setups for accurate event detection.
* **Pose estimation** (OpenPose, MediaPipe) has been applied to track body movements in combat sports.
* **Temporal models** (CNNs, Transformers) have been used for action recognition in datasets like Kinetics-400.
* **Blade tracking** parallels object tracking in fast-moving sports equipment (e.g., baseball, cricket).

Although no prior work has directly applied these techniques to fencing, combining pose estimation, object tracking, and temporal action classification forms a strong foundation for this project.

---

## Hypothesis & Expected Outcome

Fencing exchanges can be **decomposed into structured temporal sequences** of atomic actions using pose and blade trajectory features. Modeling these sequences should allow automation of right-of-way reasoning to a degree comparable to human referees.

**Expected Outcome:**
A pipeline that takes match footage as input and outputs:

1. Structured action sequences.
2. Preliminary right-of-way decisions (point assignment).

---

## Project Approach

The project follows a **three-phase pipeline**:

1. **Detection & Tracking**

   * Pose estimation (MediaPipe/OpenPose) for skeleton joints.
   * YOLO-based detection for blade tips and guards.

2. **Action Recognition**

   * Temporal classifiers (1D CNNs, Transformers, or GNNs on skeletons) to identify actions such as attack, parry, riposte, etc.
   * Each action is annotated with a **success flag** indicating whether it landed.

3. **Right-of-Way Reasoning**

   * Rule-based engine implementing FIE priority rules on recognized action sequences.
   * Future extension: Learnable sequence model for automated point calls.

---

## Label Set

### Offensive Actions

* Preparation
* Attack (Simple)
* Attack (Compound)
* Beat Attack

### Defensive Actions

* Parry
* Riposte
* Stop-cut
* Distance Pull
* Point in Line

### Other

* Recovery

**Attribute:** `success = true/false` indicates whether offensive actions landed or defensive actions were successful at avoiding the hit.

---

## Dataset Setup

### Quick Start

1. **Install dependencies**

```bash
sudo apt-get update
sudo apt-get install ffmpeg
pip install -r requirements.txt
```

2. **Download raw bout videos**

```bash
python downloader.py
```

3. **Install LosslessCut**

* Download from [LosslessCut](https://losslesscut.en.softonic.com) and extract/install.

4. **Trim and label exchanges**

* Open each raw bout video in LosslessCut.
* Split into segments:

  * Start 1 frame before fencers move.
  * End 1 frame after scorebox lights up (wait for second light if needed).
* Label segments as `Exchange_Fencer` (e.g., `1_Left`, `2_Right`).
* Export timestamps (CSV) to the `timestamps/` folder.

5. **Split raw videos into clips**

```bash
python splitter.py
```

---

## Project Workflow

1. Extract frames from video clips.
2. Run **pose and blade detection** → structured trajectory data.
3. Annotate/correct actions and success flags in CVAT.
4. Train models: detection → action recognition → right-of-way reasoning.
5. Evaluate using ground-truth referee calls.

---

## Experimental Setup

* **Datasets:** Practice or competition footage, annotated with actions and outcomes.
* **Evaluation Metrics:**

  * **Tracking Accuracy:** PCK metric, pixel error for blade tip.
  * **Action Recognition:** Precision, Recall, F1-score per action.
  * **Right-of-Way Decisions:** Agreement % with referee ground truth.

---

## Implementation Plan

1. Gather fencing footage and research vision models.
2. Process footage through models to extract skeletons and blade tips.
3. Manually adjust annotations in CVAT and label fencing actions + outcomes.
4. Build data pipeline and train action recognition models.
5. Implement FIE ruleset for point assignment and integrate into evaluation pipeline.

---

