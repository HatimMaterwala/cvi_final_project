

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(csv_path='driving_log.csv'):
    """
    Load driving_log.csv and return (image_paths, steering_angles).
    Uses ALL 3 cameras (center, left, right) to triple the dataset.
    Left/right images get a steering correction to teach recovery.
    Requirement 1.1: parse camera image paths and steering angle columns.
    """
    STEERING_CORRECTION = 0.2  # offset for left/right cameras

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find '{csv_path}'. "
            "Make sure the driving log is in the working directory."
        )

    df = pd.read_csv(csv_path, header=None)
    # Columns: 0=center, 1=left, 2=right, 3=steering, 4=throttle, 5=brake, 6=speed

    all_paths = []
    all_angles = []

    for _, row in df.iterrows():
        center_path = str(row[0]).strip()
        left_path = str(row[1]).strip()
        right_path = str(row[2]).strip()
        steering = float(row[3])

        # Center camera — use steering as-is
        all_paths.append(center_path)
        all_angles.append(steering)

        # Left camera — steer RIGHT to correct (add offset)
        all_paths.append(left_path)
        all_angles.append(steering + STEERING_CORRECTION)

        # Right camera — steer LEFT to correct (subtract offset)
        all_paths.append(right_path)
        all_angles.append(steering - STEERING_CORRECTION)

    return np.array(all_paths), np.array(all_angles, dtype=np.float32)


def plot_histogram(steering_angles, bins=25, title='Steering Angle Distribution'):
    """
    Plot a histogram of steering angle distribution.
    Requirement 1.2: display histogram to verify balance.
    """
    plt.figure(figsize=(8, 4))
    plt.hist(steering_angles, bins=bins, color='steelblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Steering Angle')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def balance_data(image_paths, steering_angles, bins=25, samples_per_bin=2000):
    """
    Trim over-represented steering angle bins so no bin exceeds samples_per_bin.
    Requirement 1.3: allow identification and trimming of over-represented bins.
    Returns balanced (image_paths, steering_angles) arrays.
    """
    bin_edges = np.histogram(steering_angles, bins=bins)[1]
    keep_indices = []

    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include right edge only for the last bin
        if i < bins - 1:
            mask = np.where((steering_angles >= lo) &
                            (steering_angles < hi))[0]
        else:
            mask = np.where((steering_angles >= lo) &
                            (steering_angles <= hi))[0]

        if len(mask) > samples_per_bin:
            mask = np.random.choice(mask, samples_per_bin, replace=False)

        keep_indices.extend(mask.tolist())

    keep_indices = np.array(keep_indices)
    return image_paths[keep_indices], steering_angles[keep_indices]
