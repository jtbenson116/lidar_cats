#!/usr/bin/env python3
"""
RPLidar Cat Detector - Step 2: Train Model
==========================================
Run this after collecting data with collect.py

Usage:
    python train.py

Output:
    lidar_model.pkl   - trained classifier
    lidar_classes.pkl - class label encoder
    training_report.txt
"""

import os
import pickle
import numpy as np
from collections import Counter

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    import sklearn
    SKLEARN_OK = True
except ImportError:
    print("ERROR: scikit-learn not installed.")
    print("Install with: pip install scikit-learn numpy")
    exit(1)

DATA_DIR = "lidar_data"
MODEL_FILE = "lidar_model.pkl"
CLASSES_FILE = "lidar_classes.pkl"


# ── Feature extraction ────────────────────────────────────────────────────────

def polar_to_cartesian(scan):
    angles = np.radians([m[1] for m in scan])
    dists = np.array([m[2] for m in scan]) / 1000.0  # mm → metres
    x = dists * np.cos(angles)
    y = dists * np.sin(angles)
    return np.column_stack((x, y))


def get_clusters(pts, eps=0.15, min_samples=3):
    """Simple distance-based clustering (no sklearn dependency at runtime)."""
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
    """12 features per cluster describing its shape and size."""
    xs, ys = cluster[:, 0], cluster[:, 1]
    width = xs.max() - xs.min()
    depth = ys.max() - ys.min()
    centroid = cluster.mean(axis=0)
    dist_from_origin = np.linalg.norm(centroid)

    # Elongation via covariance eigenvalues
    if len(cluster) > 2:
        cov = np.cov(cluster.T)
        eigvals = np.sort(np.abs(np.linalg.eigvals(cov)))
        elongation = eigvals[-1] / (eigvals[0] + 1e-6)
    else:
        elongation = 1.0

    # Angular spread
    angles = np.arctan2(ys, xs)
    angle_spread = angles.max() - angles.min()

    return [
        len(cluster),                        # 0: point count
        width,                               # 1: bounding width (m)
        depth,                               # 2: bounding depth (m)
        max(width, depth),                   # 3: max span
        min(width, depth),                   # 4: min span
        width / (depth + 1e-6),             # 5: aspect ratio
        elongation,                          # 6: elongation
        np.std(xs),                          # 7: x spread
        np.std(ys),                          # 8: y spread
        dist_from_origin,                    # 9: distance from lidar
        angle_spread,                        # 10: angular spread
        len(cluster) / (width * depth + 1e-6),  # 11: point density
    ]


def scan_to_features(scan):
    """
    Convert a full scan into a feature vector.
    We take the top-3 clusters by size and describe each one.
    Returns a flat vector of 36 features (3 clusters × 12 features).
    """
    pts = polar_to_cartesian(scan)
    clusters = get_clusters(pts)

    # Sort by point count, keep up to 3
    clusters = sorted(clusters, key=len, reverse=True)[:3]

    features = []
    for cluster in clusters:
        features.extend(extract_cluster_features(cluster))

    # Pad to 36 features if fewer than 3 clusters
    while len(features) < 36:
        features.extend([0.0] * 12)

    return features[:36]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    all_features = []
    all_labels = []

    if not os.path.exists(DATA_DIR):
        print(f"ERROR: No data directory '{DATA_DIR}' found.")
        print("Run collect.py first!")
        exit(1)

    pkl_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pkl")]
    if not pkl_files:
        print(f"ERROR: No .pkl files found in '{DATA_DIR}'")
        print("Run collect.py first!")
        exit(1)

    print(f"Loading data from {len(pkl_files)} file(s)...\n")

    label_counts = Counter()
    for fname in sorted(pkl_files):
        path = os.path.join(DATA_DIR, fname)
        with open(path, "rb") as f:
            records = pickle.load(f)

        for record in records:
            feats = scan_to_features(record["scan"])
            all_features.append(feats)
            all_labels.append(record["label"])
            label_counts[record["label"]] += 1

        print(f"  ✓ {fname}  ({len(records)} scans)")

    print(f"\nClass distribution:")
    for label, count in sorted(label_counts.items()):
        bar = "█" * min(40, count // 5)
        print(f"  {label:<10} {bar:<40} {count}")

    return np.array(all_features), np.array(all_labels)


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    print("=" * 55)
    print("  RPLidar Object Classifier - Training")
    print("=" * 55 + "\n")

    X, y = load_data()

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"\nClasses: {list(le.classes_)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}\n")

    # Train Random Forest (fast, interpretable, good on small data)
    print("Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = (y_pred == y_test).mean()
    print(f"\nTest accuracy: {acc:.1%}\n")

    # Cross-validation
    cv_scores = cross_val_score(clf, X, y_enc, cv=5, scoring="accuracy")
    print(f"5-fold CV accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}\n")

    # Detailed report
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print("Classification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    header = "         " + "  ".join(f"{c[:6]:>6}" for c in le.classes_)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {le.classes_[i]:<8} " + "  ".join(f"{v:>6}" for v in row))

    # Feature importance
    print("\nTop 5 most important features:")
    feat_names = []
    for i in range(3):
        for name in ["points", "width", "depth", "max_span", "min_span",
                     "aspect", "elongation", "std_x", "std_y", "dist", "angle_spread", "density"]:
            feat_names.append(f"cluster{i+1}_{name}")
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:5]
    for idx in top_idx:
        print(f"  {feat_names[idx]:<30} {importances[idx]:.3f}")

    # Save model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)
    with open(CLASSES_FILE, "wb") as f:
        pickle.dump(le, f)

    print(f"\n✓ Model saved to {MODEL_FILE}")
    print(f"✓ Classes saved to {CLASSES_FILE}")

    # Save report
    with open("training_report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.1%}\n")
        f.write(f"CV: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}\n\n")
        f.write(report)
        f.write(f"\nClasses: {list(le.classes_)}\n")
    print("✓ Report saved to training_report.txt\n")

    if acc < 0.7:
        print("⚠  Accuracy below 70%. Tips:")
        print("   - Collect more data (300+ scans per class)")
        print("   - Make sure scenes are distinct during recording")
        print("   - Check for class imbalance above")
    elif acc >= 0.85:
        print("🎉 Great accuracy! Model is ready for live detection.")
    else:
        print("✓ Decent accuracy. More data will improve it further.")


if __name__ == "__main__":
    train()
