"""
data_preprocessing.py
CVI620 Final Project — Winter 2026
Self-Driving Car Simulation using CNN

Responsibilities:
  - Load driving_log.csv and extract center-camera image paths + steering angles
  - Plot and analyse the steering angle distribution (histogram)
  - Balance the dataset by capping over-represented near-zero steering bins
  - Split into training and validation sets (80 / 20)
  - Provide preprocess_image(): crop → YUV → resize 200×66 → Gaussian blur → normalize
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "driving_log.csv")

# Balancing parameters
NUM_BINS = 25          # number of histogram bins
SAMPLES_PER_BIN = 1000 # maximum samples allowed per bin after balancing

# Image dimensions expected by the Nvidia model
IMG_HEIGHT = 66
IMG_WIDTH  = 200

# Crop rows (from Figure 6 in the project spec)
CROP_TOP    = 60
CROP_BOTTOM = 135


# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
def load_data(csv_path: str = CSV_PATH, data_dir: str = DATA_DIR):
    """
    Load the driving log CSV and return arrays of image paths and steering angles.

    The CSV columns are:
        center, left, right, steering, throttle, brake, speed
    Only the center camera and steering are used for this project.

    Returns
    -------
    image_paths : np.ndarray of str
    steering    : np.ndarray of float32
    """
    print(f"[INFO] Loading CSV from: {csv_path}")
    columns = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
    df = pd.read_csv(csv_path, names=columns)

    # Strip whitespace from path strings
    df["center"] = df["center"].str.strip()

    # Build absolute paths (handle both absolute paths saved by simulator and
    # relative paths where only the filename is stored)
    def resolve_path(p: str) -> str:
        if os.path.isabs(p) and os.path.exists(p):
            return p
        # Try relative to data/IMG/
        basename = os.path.basename(p)
        candidate = os.path.join(data_dir, "IMG", basename)
        if os.path.exists(candidate):
            return candidate
        # Fall back to path as-is
        return p

    df["center"] = df["center"].apply(resolve_path)

    image_paths = df["center"].values
    steering    = df["steering"].values.astype(np.float32)

    print(f"[INFO] Total samples loaded: {len(image_paths)}")
    return image_paths, steering


# ─────────────────────────────────────────────
# 2. Plot Steering Histogram
# ─────────────────────────────────────────────
def plot_histogram(steering: np.ndarray, title: str = "Steering Angle Distribution",
                   limit_line: int = None, show: bool = True):
    """
    Plot a histogram of steering angle values.

    Parameters
    ----------
    steering    : array of steering angles
    title       : plot title
    limit_line  : optional horizontal threshold line (samples_per_bin cap)
    show        : whether to call plt.show()
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(steering, bins=NUM_BINS, color="#2196F3", edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Steering Angle")
    ax.set_ylabel("Count")
    ax.set_xlim(-1.05, 1.05)

    if limit_line is not None:
        ax.axhline(y=limit_line, color="cyan", linewidth=1.5,
                   linestyle="--", label=f"Cap = {limit_line}")
        ax.legend()

    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────
# 3. Balance Dataset
# ─────────────────────────────────────────────
def balance_dataset(image_paths: np.ndarray, steering: np.ndarray,
                    samples_per_bin: int = SAMPLES_PER_BIN,
                    num_bins: int = NUM_BINS):
    """
    Remove excess samples from over-represented steering bins so that no bin
    has more than `samples_per_bin` entries.  This prevents the model from
    learning a strong bias toward driving straight.

    Returns
    -------
    image_paths_bal : balanced image path array
    steering_bal    : balanced steering array
    """
    print(f"[INFO] Balancing dataset (cap={samples_per_bin} per bin) …")
    bin_edges = np.linspace(-1, 1, num_bins + 1)
    keep_indices = []

    for i in range(num_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        # find indices whose steering falls in this bin
        bin_idx = np.where((steering >= low) & (steering < high))[0]
        if len(bin_idx) > samples_per_bin:
            bin_idx = np.random.choice(bin_idx, samples_per_bin, replace=False)
        keep_indices.extend(bin_idx.tolist())

    keep_indices = np.array(keep_indices)
    image_paths_bal = image_paths[keep_indices]
    steering_bal    = steering[keep_indices]

    print(f"[INFO] Samples after balancing: {len(image_paths_bal)}")
    return image_paths_bal, steering_bal


# ─────────────────────────────────────────────
# 4. Train / Validation Split
# ─────────────────────────────────────────────
def split_data(image_paths: np.ndarray, steering: np.ndarray,
               test_size: float = 0.2, random_state: int = 42):
    """
    Shuffle and split data into training and validation sets.

    Returns
    -------
    X_train, X_val, y_train, y_val
    """
    image_paths, steering = shuffle(image_paths, steering, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, steering,
        test_size=test_size,
        random_state=random_state
    )
    print(f"[INFO] Training samples  : {len(X_train)}")
    print(f"[INFO] Validation samples: {len(X_val)}")
    return X_train, X_val, y_train, y_val


# ─────────────────────────────────────────────
# 5. Image Preprocessing Pipeline
# ─────────────────────────────────────────────
def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Apply the full preprocessing pipeline to a single BGR image:

      1. Crop: keep rows [60, 135] to remove sky and hood (Figure 6)
      2. Convert BGR → YUV colour space (as used by Nvidia)
      3. Apply Gaussian blur to reduce noise
      4. Resize to 200×66 pixels (Nvidia model input)
      5. Normalise pixel values to [-1, 1]

    Parameters
    ----------
    img : np.ndarray  — BGR image as loaded by cv2.imread()

    Returns
    -------
    np.ndarray of shape (66, 200, 3), dtype float32, values in [-1, 1]
    """
    # Step 1: Crop road region
    img = img[CROP_TOP:CROP_BOTTOM, :, :]

    # Step 2: BGR → YUV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # Step 3: Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Step 4: Resize to Nvidia input size (width × height)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    # Step 5: Normalize to [-1, 1]
    img = img.astype(np.float32) / 127.5 - 1.0

    return img


def load_and_preprocess_image(path: str) -> np.ndarray:
    """Load an image from disk and run the full preprocessing pipeline."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return preprocess_image(img)


# ─────────────────────────────────────────────
# Quick Self-Test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Verify CSV exists
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] driving_log.csv not found at {CSV_PATH}")
        print("  → Place your data/ folder (with IMG/ and driving_log.csv) in the project root.")
        raise SystemExit(1)

    # 1. Load
    paths, angles = load_data()

    # 2. Plot raw distribution
    print("[INFO] Plotting raw steering distribution …")
    plot_histogram(angles, title="Raw Steering Angle Distribution",
                   limit_line=SAMPLES_PER_BIN)

    # 3. Balance
    paths_bal, angles_bal = balance_dataset(paths, angles)

    # 4. Plot balanced distribution
    print("[INFO] Plotting balanced steering distribution …")
    plot_histogram(angles_bal, title="Balanced Steering Angle Distribution")

    # 5. Split
    X_train, X_val, y_train, y_val = split_data(paths_bal, angles_bal)

    # 6. Quick image preprocess sanity check
    print("[INFO] Running preprocessing sanity check on first training image …")
    sample = load_and_preprocess_image(X_train[0])
    print(f"[INFO] Preprocessed image shape: {sample.shape}, "
          f"min={sample.min():.3f}, max={sample.max():.3f}")
    print("[PASS] data_preprocessing.py — all checks passed.")
