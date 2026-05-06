#!/usr/bin/env python3
"""
RPLidar Cat Detector - Step 1: Data Collection
=============================================
Run this script to record labeled lidar scans for training.

Usage:
    python collect.py --label cat --scans 300
    python collect.py --label empty --scans 300
    python collect.py --label chair --scans 300
    python collect.py --label person --scans 300

Labels to collect (suggested order):
  1. empty    - room with nothing interesting in it
  2. cat      - your cat sitting/walking in view
  3. chair    - a chair or small furniture
  4. person   - a person standing/sitting nearby

Tips:
  - Move your cat around during recording to get varied poses
  - Collect at least 200-300 scans per class
  - Run multiple sessions with different cat positions
"""

import argparse
import pickle
import os
import time
from datetime import datetime

try:
    from rplidar import RPLidar
    LIDAR_AVAILABLE = True
except ImportError:
    print("WARNING: rplidar not installed. Running in DEMO mode with fake data.")
    print("Install with: pip install rplidar-roboticia\n")
    LIDAR_AVAILABLE = False

DATA_DIR = "lidar_data"
PORT = "/dev/ttyUSB0"  # Change to /dev/ttyAMA0 if using GPIO UART


def fake_scan(label):
    """Generate fake scan data for testing without hardware."""
    import random
    import math
    scan = []
    for i in range(360):
        angle = float(i)
        if label == "cat":
            # Small blob at ~1m in front
            if 80 <= i <= 100:
                dist = 1000 + random.gauss(0, 20)
            else:
                dist = 3000 + random.gauss(0, 50)
        elif label == "person":
            # Two leg clusters
            if 85 <= i <= 95 or 105 <= i <= 115:
                dist = 1500 + random.gauss(0, 30)
            else:
                dist = 3000 + random.gauss(0, 50)
        elif label == "chair":
            # 4 legs
            for leg_angle in [80, 95, 105, 120]:
                if abs(i - leg_angle) <= 3:
                    dist = 1200 + random.gauss(0, 15)
                    break
            else:
                dist = 3000 + random.gauss(0, 50)
        else:  # empty
            dist = 3000 + random.gauss(0, 50)
        scan.append((90, angle, max(100, dist)))
    return scan


def collect(label, num_scans, port):
    os.makedirs(DATA_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(DATA_DIR, f"{label}_{timestamp}.pkl")

    print(f"\n{'='*50}")
    print(f"  Collecting: {label.upper()}")
    print(f"  Target scans: {num_scans}")
    print(f"  Output: {filename}")
    print(f"{'='*50}")
    print(f"\nGet your scene ready, then press ENTER to start...")
    input()

    scans = []

    if LIDAR_AVAILABLE:
        lidar = RPLidar(port)
        try:
            print(f"Starting collection... (Ctrl+C to stop early)\n")
            for i, scan in enumerate(lidar.iter_scans()):
                scans.append({
                    "scan": scan,
                    "label": label,
                    "timestamp": time.time()
                })
                # Progress bar
                pct = int((i + 1) / num_scans * 40)
                bar = "█" * pct + "░" * (40 - pct)
                print(f"\r  [{bar}] {i+1}/{num_scans}", end="", flush=True)
                if i + 1 >= num_scans:
                    break
        except KeyboardInterrupt:
            print(f"\n  Stopped early at {len(scans)} scans.")
        finally:
            lidar.stop()
            lidar.disconnect()
    else:
        # Demo mode
        print(f"DEMO MODE: Generating {num_scans} fake '{label}' scans...\n")
        for i in range(num_scans):
            scans.append({
                "scan": fake_scan(label),
                "label": label,
                "timestamp": time.time()
            })
            pct = int((i + 1) / num_scans * 40)
            bar = "█" * pct + "░" * (40 - pct)
            print(f"\r  [{bar}] {i+1}/{num_scans}", end="", flush=True)
            time.sleep(0.01)

    print(f"\n\n  ✓ Saved {len(scans)} scans to {filename}")

    with open(filename, "wb") as f:
        pickle.dump(scans, f)

    # Show summary of all collected data
    print(f"\n--- All collected data in '{DATA_DIR}/' ---")
    counts = {}
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".pkl"):
            lbl = fname.split("_")[0]
            counts[lbl] = counts.get(lbl, 0)
            with open(os.path.join(DATA_DIR, fname), "rb") as f:
                counts[lbl] += len(pickle.load(f))
    for lbl, count in sorted(counts.items()):
        bar = "█" * min(40, count // 10)
        status = "✓ Ready" if count >= 200 else f"⚠ Need {200-count} more"
        print(f"  {lbl:<10} {bar:<40} {count:>4} scans  {status}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect lidar scan data for training")
    parser.add_argument("--label", required=True,
                        choices=["cat", "empty", "chair", "person", "other"],
                        help="Label for this recording session")
    parser.add_argument("--scans", type=int, default=300,
                        help="Number of scans to collect (default: 300)")
    parser.add_argument("--port", default=PORT,
                        help=f"Serial port (default: {PORT})")
    args = parser.parse_args()

    collect(args.label, args.scans, args.port)
