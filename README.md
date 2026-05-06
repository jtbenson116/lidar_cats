# RPLidar A1 Cat Detector - WIP: results pending
### Raspberry Pi 5 · Complete Setup Guide

---

## What You're Getting

Three scripts that work as a pipeline:

```
collect.py  →  train.py  →  detect.py
  (record scans)       (train model)        (live detection)
```

---

## 1. Install Dependencies

```bash
pip install rplidar-roboticia scikit-learn numpy matplotlib
```

---

## 2. Connect Your Lidar

Plug the RPLidar A1 USB adapter into your Pi. Check it shows up:

```bash
ls /dev/ttyUSB*   # should show /dev/ttyUSB0
```

Add yourself to the dialout group (do this once, then log out/in):

```bash
sudo usermod -a -G dialout $USER
```

---

## 3. Collect Training Data

Run the collector for each object type. **Move your cat around** during
recording to capture different poses.

```bash
# Step 1: empty room (no cat, no people)
python collect.py --label empty --scans 300

# Step 2: cat in view (move it around!)
python collect.py --label cat --scans 300

# Step 3: a chair or table
python collect.py --label chair --scans 300

# Step 4: a person standing/sitting
python collect.py --label person --scans 300
```

Minimum recommended: **200 scans per class**.
More is better — run multiple sessions with your cat in different positions.

Your scans are saved in `lidar_data/` as .pkl files.

---

## 4. Train the Model

```bash
python train.py
```

This reads all your collected scans, extracts features, trains a
Random Forest classifier, and shows you accuracy metrics.

Expect: 70-85% accuracy. The `cat` class will be the hardest.

Outputs: `lidar_model.pkl`, `lidar_classes.pkl`, `training_report.txt`

---

## 5. Run Live Detection

```bash
python detect.py
```

A dark window opens showing your lidar sweep in real-time with
colored bounding boxes and labels on detected objects.

**Orange = cat detected!** 🐱

---

## Tips for Better Cat Detection

| Problem | Fix |
|---------|-----|
| Cat not detected | Lidar beam may be too high/low — adjust height |
| Too many false positives | Collect more "empty" scans |
| Low confidence | Collect more cat scans in varied poses |
| Cat looks like chair | Collect scans of both side by side |

### Ideal lidar height
Mount the lidar at **20–30cm** off the ground. This catches a sitting
or walking cat reliably. A lying-flat cat may still be missed.

---

## Testing Without Hardware

All three scripts support demo mode — no lidar needed:

```bash
python collect.py --label cat --scans 100   # generates fake data
python train.py                             # trains on fake data
python detect.py --demo                      # animated demo
```

---

## File Structure

```
your-project/
├── collect.py
├── train.py
├── detect.py
├── README.md
├── lidar_data/           ← created when you collect data
│   ├── cat_20260506_120000.pkl
│   ├── empty_20260506_121500.pkl
│   └── ...
├── lidar_model.pkl       ← created after training
├── lidar_classes.pkl     ← created after training
└── training_report.txt   ← created after training
```

---

## Want Better Accuracy?

Add a Pi Camera + YOLO to confirm cat detections:

```bash
pip install ultralytics opencv-python
```

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # tiny model, runs ~5fps on Pi 5
# class 15 in COCO is 'cat'
results = model(frame)
cats = [r for r in results[0].boxes if r.cls == 15]
```

The lidar gives you **position and distance**, the camera gives you
**"yes that's definitely a cat"** — together they're very reliable.
