# Automated Refereeing for Olympic Saber

---

## Project Overview

This project aims to develop a computer vision system capable of **analyzing Olympic-style saber fencing bouts** and assisting referees in determining **right-of-way** and point assignment. The system is structured into three phases:

1. **Detection & Tracking**

   * Detect fencer poses (skeleton joints) and blade positions using MediaPipe/OpenPose and YOLOv8.
   * Extract structured features such as joint coordinates, blade tip positions, and derived metrics like velocities.

2. **Action Recognition**

   * Classify fencing actions (attack, parry, riposte, etc.) using temporal models like 1D CNNs, Transformers, or Graph Neural Networks on skeleton data.
   * Each action may be annotated with a **success flag** indicating whether it landed.

3. **Right-of-Way Reasoning**

   * Apply FIE priority rules to sequences of recognized actions to determine points awarded.
   * Initially rule-based, with the option to train a learnable sequence model for automatic decision-making.

---

## Quick Start – Prepare Dataset

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
  * End 1 frame after scorebox lights up (wait for 2nd light if needed).
* Label segments as `Exchange_Fencer` (e.g., `1_Left`, `2_Right`).
* Export timestamps (CSV) to the `timestamps/` folder.

5. **Split raw videos into clips**

```bash
python splitter.py
```

* Clips are saved in `clips/` folder, ready for annotation and processing.

---

## Label Set

### Offensive Actions

* Preparation
* Attack (Simple)
* Attack (Compound)
* Beat Attack
* Remise
* Counter-attack
* Stop-cut

### Defensive Actions

* Parry
* Riposte
* Counter-riposte
* Void (distance pull, duck, lean, etc.)

### Special Actions

* Point in Line
* Simultaneous Attack
* No Action / Reset

**Notes:**

* Each action can be annotated with a `success` attribute (`true`/`false`) to indicate whether the action landed successfully.
* Labels are actor-specific (Left / Right fencer).

---

## Dataset Setup

### **1. Environment Setup**

```bash
# Install ffmpeg
sudo apt-get update
sudo apt-get install ffmpeg

# Install Python dependencies
pip install -r requirements.txt
```

### **2. Download Raw Videos**

```bash
python downloader.py
```

*(This script will download all raw bout videos to the `raw/` folder.)*

### **3. Download & Install LosslessCut**

* Visit [LosslessCut](https://losslesscut.en.softonic.com)
* Extract and install to a local directory. This tool will be used for trimming and labeling video segments.

---

## Trimming and Splitting Videos

1. Open a raw bout video in LosslessCut.
2. Split the video into segments by **exchange**:

   * Start the segment **1 frame before both fencers move**.
   * End the segment **1 frame after the scorebox lights up** (if both lights go off, wait for the second light).
3. Label each segment with the exchange number and point winner, e.g., `1_Left`, `2_Right`, etc.
4. After labeling all segments in a bout:

   * Go to `File -> Export Project -> Timestamps (CSV)`.
   * Rename the file to the bout ID (e.g., `BOUT12.csv`) and save it to the `timestamps/` folder.
5. After labeling all bouts, split the raw videos into individual clips:

```bash
python splitter.py
```

*(This will generate per-exchange clip files in the `clips/` folder.)*

---

## Project Workflow

1. **Video → Frames**: Extract frames from clips.
2. **Pose & Blade Detection**: Run MediaPipe/OpenPose + YOLOv8 to extract skeletons and blade tips.
3. **Annotation**: Correct detections and label actions, actor, and success attributes.
4. **Action Recognition**: Train temporal models on annotated sequences.
5. **Right-of-Way Reasoning**: Apply rule-based or learned sequence logic to produce referee decisions.
6. **Evaluation**:

   * **Tracking:** PCK metric, blade tip pixel error.
   * **Action Recognition:** Precision, Recall, F1.
   * **Right-of-Way Decisions:** Agreement with referee ground truth.