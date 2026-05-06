#!/usr/bin/env python3
"""
RPLidar Cat Detector - Step 3: Live Detection
=============================================
Run this after training with train.py.
Opens a live matplotlib window showing the lidar scan
with detected objects labeled in real-time.

Usage:
    python detect.py
    python detect.py --port /dev/ttyAMA0   # for GPIO UART
    python detect.py --demo                  # no hardware needed
"""

import argparse
import pickle
import time
import os
import numpy as np
from collections import deque

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_OK = True
except ImportError:
    print("ERROR: matplotlib not installed. pip install matplotlib")
    exit(1)

try:
    from rplidar import RPLidar
    LIDAR_AVAILABLE = True
except ImportError:
    LIDAR_AVAILABLE = False

MODEL_FILE = "lidar_model.pkl"
CLASSES_FILE = "lidar_classes.pkl"
PORT = "/dev/ttyUSB0"

# Colors per class
CLASS_COLORS = {
    "cat":    "#FF6B35",
    "person": "#4ECDC4",
    "chair":  "#95E1D3",
    "empty":  "#A8DADC",
    "other":  "#AAAAAA",
}
UNKNOWN_COLOR = "#FFFFFF"


# ── Feature extraction (same as training) ─────────────────────────────────────

def polar_to_cartesian(scan):
    angles = np.radians([m[1] for m in scan])
    dists = np.array([m[2] for m in scan]) / 1000.0
    x = dists * np.cos(angles)
    y = dists * np.sin(angles)
    return np.column_stack((x, y))


def get_clusters(pts, eps=0.15, min_samples=3):
    if len(pts) < min_samples:
        return []
    visited = set()
    clusters = []

    def neighbors(idx):
        diffs = pts - pts[idx]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        return list(np.where(dists < eps)[0])

    for i in range(len(pts)):
        if i in visited:
            continue
        nbrs = neighbors(i)
        if len(nbrs) < min_samples:
            continue
        cluster = set(nbrs)
        queue = list(nbrs)
        while queue:
            q = queue.pop()
            if q in visited:
                continue
            visited.add(q)
            q_nbrs = neighbors(q)
            if len(q_nbrs) >= min_samples:
                new = set(q_nbrs) - cluster
                cluster |= new
                queue.extend(new)
        clusters.append(pts[list(cluster)])
    return clusters


def extract_cluster_features(cluster):
    xs, ys = cluster[:, 0], cluster[:, 1]
    width = xs.max() - xs.min()
    depth = ys.max() - ys.min()
    centroid = cluster.mean(axis=0)
    dist_from_origin = np.linalg.norm(centroid)
    if len(cluster) > 2:
        cov = np.cov(cluster.T)
        eigvals = np.sort(np.abs(np.linalg.eigvals(cov)))
        elongation = eigvals[-1] / (eigvals[0] + 1e-6)
    else:
        elongation = 1.0
    angles = np.arctan2(ys, xs)
    angle_spread = angles.max() - angles.min()
    return [
        len(cluster), width, depth, max(width, depth), min(width, depth),
        width / (depth + 1e-6), elongation, np.std(xs), np.std(ys),
        dist_from_origin, angle_spread, len(cluster) / (width * depth + 1e-6),
    ]


def scan_to_features(scan):
    pts = polar_to_cartesian(scan)
    clusters = get_clusters(pts)
    clusters = sorted(clusters, key=len, reverse=True)[:3]
    features = []
    for cluster in clusters:
        features.extend(extract_cluster_features(cluster))
    while len(features) < 36:
        features.extend([0.0] * 12)
    return features[:36], clusters


# ── Fake scan for demo mode ────────────────────────────────────────────────────

def fake_scan_generator():
    """Generates a rotating fake scan with a moving cat blob."""
    import math, random
    t = 0
    while True:
        scan = []
        cat_angle = 90 + 20 * math.sin(t * 0.05)
        for i in range(360):
            angle = float(i)
            diff = abs(i - cat_angle)
            if diff < 10:
                dist = 1000 + random.gauss(0, 25) + diff * 5
            else:
                dist = 3000 + random.gauss(0, 60)
            scan.append((90, angle, max(150, dist)))
        t += 1
        yield scan


# ── Main visualizer ────────────────────────────────────────────────────────────

class LiveDetector:
    def __init__(self, clf, le, port, demo=False):
        self.clf = clf
        self.le = le
        self.port = port
        self.demo = demo

        self.current_scan = []
        self.scan_gen = None
        self.lidar = None

        # Detection history for smoothing
        self.detection_history = deque(maxlen=5)
        self.fps_times = deque(maxlen=20)
        self.frame_count = 0

        self._setup_lidar()
        self._setup_plot()

    def _setup_lidar(self):
        if self.demo or not LIDAR_AVAILABLE:
            print("Running in DEMO mode (no hardware)")
            self.scan_gen = fake_scan_generator()
        else:
            print(f"Connecting to lidar on {self.port}...")
            self.lidar = RPLidar(self.port)
            self.scan_gen = self.lidar.iter_scans()
            print("Connected!")

    def _setup_plot(self):
        plt.style.use("dark_background")
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.fig.patch.set_facecolor("#0A0A0F")
        self.ax.set_facecolor("#0A0A0F")

        # Grid rings
        for r in [1, 2, 3, 4, 5]:
            circle = plt.Circle((0, 0), r, fill=False,
                                 color="#1A1A2E", linewidth=0.8, linestyle="--")
            self.ax.add_patch(circle)
            self.ax.text(0, r + 0.05, f"{r}m", color="#333355",
                         fontsize=7, ha="center", va="bottom")

        # Crosshair
        self.ax.axhline(0, color="#1A1A2E", linewidth=0.5)
        self.ax.axvline(0, color="#1A1A2E", linewidth=0.5)

        self.ax.set_xlim(-5.5, 5.5)
        self.ax.set_ylim(-5.5, 5.5)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Lidar origin marker
        self.ax.plot(0, 0, "o", color="#00FFAA", markersize=8, zorder=10)
        self.ax.text(0.1, 0.1, "LIDAR", color="#00FFAA",
                     fontsize=8, fontweight="bold")

        # Title
        self.title = self.ax.set_title(
            "RPLidar Live Detection",
            color="#FFFFFF", fontsize=14, fontweight="bold", pad=15
        )

        # Legend
        legend_patches = [
            mpatches.Patch(color=v, label=k)
            for k, v in CLASS_COLORS.items()
            if k in self.le.classes_
        ]
        self.ax.legend(handles=legend_patches, loc="upper right",
                       fontsize=9, framealpha=0.3, facecolor="#111")

        # FPS counter
        self.fps_text = self.ax.text(
            -5.3, -5.2, "", color="#555577", fontsize=8
        )

        # Status box
        self.status_text = self.ax.text(
            -5.3, 5.0, "", color="#FFFFFF", fontsize=11,
            fontweight="bold", verticalalignment="top"
        )

        plt.tight_layout()

    def _smooth_detections(self, detections):
        """Vote across recent frames to reduce flicker."""
        self.detection_history.append(detections)
        if not self.detection_history:
            return detections

        # Just return current for now (could add voting logic)
        return detections

    def update(self, frame):
        try:
            scan = next(self.scan_gen)
        except StopIteration:
            return

        self.current_scan = scan
        now = time.time()
        self.fps_times.append(now)

        # Clear dynamic elements
        for artist in self.ax.collections[:]:
            artist.remove()
        for txt in self.ax.texts[3:]:  # keep first 3 static labels
            txt.remove()

        # Convert and cluster
        pts = polar_to_cartesian(scan)
        feats, clusters = scan_to_features(scan)

        # Draw all raw points (faint)
        if len(pts) > 0:
            self.ax.scatter(pts[:, 0], pts[:, 1],
                            s=1, c="#223344", alpha=0.5, zorder=1)

        # Classify and draw each cluster
        cat_detected = False
        detections = []

        for i, cluster in enumerate(clusters):
            # Build features for just this cluster (padded)
            c_feats = extract_cluster_features(cluster)
            full = c_feats + [0.0] * 24  # pad other cluster slots
            full = full[:36]

            proba = self.clf.predict_proba([full])[0]
            pred_idx = np.argmax(proba)
            label = self.le.classes_[pred_idx]
            confidence = proba[pred_idx]

            # Skip low-confidence and empty predictions for overlay
            if confidence < 0.4 or label == "empty":
                color = "#334455"
                label_text = ""
            else:
                color = CLASS_COLORS.get(label, UNKNOWN_COLOR)
                label_text = f"{label} {confidence:.0%}"
                if label == "cat":
                    cat_detected = True
                detections.append((label, confidence))

            # Draw cluster points
            self.ax.scatter(cluster[:, 0], cluster[:, 1],
                            s=15, c=color, alpha=0.9, zorder=5)

            # Draw bounding box
            if label_text:
                xs, ys = cluster[:, 0], cluster[:, 1]
                pad = 0.05
                rect = mpatches.FancyBboxPatch(
                    (xs.min() - pad, ys.min() - pad),
                    (xs.max() - xs.min()) + 2 * pad,
                    (ys.max() - ys.min()) + 2 * pad,
                    boxstyle="round,pad=0.02",
                    linewidth=1.5, edgecolor=color,
                    facecolor="none", zorder=6
                )
                self.ax.add_patch(rect)

                centroid = cluster.mean(axis=0)
                self.ax.text(
                    centroid[0], cluster[:, 1].max() + 0.15,
                    label_text, color=color, fontsize=9,
                    fontweight="bold", ha="center", zorder=7
                )

        # Status display
        if cat_detected:
            status = "🐱 CAT DETECTED"
            status_color = CLASS_COLORS["cat"]
        elif detections:
            top = max(detections, key=lambda x: x[1])
            status = f"Detected: {top[0]}"
            status_color = CLASS_COLORS.get(top[0], UNKNOWN_COLOR)
        else:
            status = "Scanning..."
            status_color = "#446688"

        self.ax.text(
            -5.3, 5.0, status, color=status_color, fontsize=12,
            fontweight="bold", verticalalignment="top", zorder=8
        )

        # FPS
        if len(self.fps_times) >= 2:
            fps = (len(self.fps_times) - 1) / (self.fps_times[-1] - self.fps_times[0])
            self.ax.text(-5.3, -5.2, f"{fps:.1f} fps  |  {len(pts)} pts  |  {len(clusters)} clusters",
                         color="#445566", fontsize=8)

        self.frame_count += 1

    def run(self):
        print("\nStarting live detection... Close the window to stop.\n")
        ani = FuncAnimation(self.fig, self.update, interval=50, cache_frame_data=False)
        plt.show()
        self.cleanup()

    def cleanup(self):
        if self.lidar:
            try:
                self.lidar.stop()
                self.lidar.disconnect()
            except Exception:
                pass
        print("Lidar stopped.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live lidar object detection")
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode without hardware")
    args = parser.parse_args()

    demo = args.demo or not LIDAR_AVAILABLE

    # Load model
    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: Model file '{MODEL_FILE}' not found.")
        print("Run train.py first!")
        exit(1)

    print("Loading model...")
    with open(MODEL_FILE, "rb") as f:
        clf = pickle.load(f)
    with open(CLASSES_FILE, "rb") as f:
        le = pickle.load(f)

    print(f"Model loaded. Classes: {list(le.classes_)}")

    detector = LiveDetector(clf, le, args.port, demo=demo)
    detector.run()


if __name__ == "__main__":
    main()
