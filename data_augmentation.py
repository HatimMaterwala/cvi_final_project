"""
data_augmentation.py
CVI620 Final Project — Winter 2026
Self-Driving Car Simulation using CNN

Applies random data augmentation to training images to improve model generalisation.
Each transformation is applied independently with ~50 % probability.

Augmentations implemented:
  - Flip          : horizontal mirror, negate steering angle
  - Brightness    : random HSV value-channel scaling
  - Pan           : random horizontal + vertical translation
  - Zoom          : random zoom-in crop
  - Rotation      : random ±15° rotation

IMPORTANT: Augmentation must only be applied to TRAINING data, not validation data.
"""

import numpy as np
import cv2


# ─────────────────────────────────────────────
# Individual Augmentation Helpers
# ─────────────────────────────────────────────

def random_flip(img: np.ndarray, steering: float):
    """
    Randomly flip the image horizontally.
    When flipped, the steering angle must be negated.

    Applied with 50 % probability.
    """
    if np.random.rand() < 0.5:
        img      = cv2.flip(img, 1)
        steering = -steering
    return img, steering


def random_brightness(img: np.ndarray, steering: float):
    """
    Randomly adjust image brightness by scaling the V-channel in HSV space.
    Does NOT affect the steering angle.

    Applied with 50 % probability.
    Scale factor drawn uniformly from [0.4, 1.2].
    """
    if np.random.rand() < 0.5:
        img_hsv         = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        scale           = np.random.uniform(0.4, 1.2)
        img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * scale, 0, 255)
        img             = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img, steering


def random_pan(img: np.ndarray, steering: float,
               max_h_shift: float = 0.15, max_v_shift: float = 0.10):
    """
    Randomly translate the image horizontally and vertically.
    Horizontal shift is also added to steering (0.004 per pixel shift).

    Applied with 50 % probability.
    """
    if np.random.rand() < 0.5:
        h, w = img.shape[:2]
        tx   = int(np.random.uniform(-max_h_shift, max_h_shift) * w)
        ty   = int(np.random.uniform(-max_v_shift, max_v_shift) * h)

        M       = np.float32([[1, 0, tx], [0, 1, ty]])
        img     = cv2.warpAffine(img, M, (w, h))
        steering += tx * 0.004   # empirical coefficient
    return img, steering


def random_zoom(img: np.ndarray, steering: float,
                zoom_range: float = 0.20):
    """
    Randomly zoom into the image by cropping a central region and resizing it
    back to the original dimensions.

    Applied with 50 % probability.
    """
    if np.random.rand() < 0.5:
        h, w    = img.shape[:2]
        zoom    = np.random.uniform(1.0, 1.0 + zoom_range)
        new_h   = int(h / zoom)
        new_w   = int(w / zoom)
        top     = (h - new_h) // 2
        left    = (w - new_w) // 2
        img     = img[top:top + new_h, left:left + new_w]
        img     = cv2.resize(img, (w, h))
    return img, steering


def random_rotation(img: np.ndarray, steering: float,
                    max_angle: float = 15.0):
    """
    Randomly rotate the image by ±max_angle degrees.
    Steering is adjusted by a small proportional factor.

    Applied with 50 % probability.
    """
    if np.random.rand() < 0.5:
        h, w  = img.shape[:2]
        angle = np.random.uniform(-max_angle, max_angle)
        M     = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img   = cv2.warpAffine(img, M, (w, h))
        # Slight steering correction for rotation
        steering += (-angle / max_angle) * 0.05
    return img, steering


# ─────────────────────────────────────────────
# Combined Augmentation Pipeline
# ─────────────────────────────────────────────

def augment_image(img: np.ndarray, steering: float):
    """
    Apply the full random augmentation pipeline to a single BGR image.

    Each individual transform is applied independently with ~50 % probability,
    so the combination of transforms applied varies per call.

    Parameters
    ----------
    img      : np.ndarray — BGR image (uint8)
    steering : float      — steering angle (will be modified in-place for flip/pan)

    Returns
    -------
    img      : np.ndarray — augmented BGR image (uint8)
    steering : float      — (possibly modified) steering angle
    """
    img, steering = random_flip(img, steering)
    img, steering = random_brightness(img, steering)
    img, steering = random_pan(img, steering)
    img, steering = random_zoom(img, steering)
    img, steering = random_rotation(img, steering)

    # Clamp steering to valid range [-1, 1]
    steering = float(np.clip(steering, -1.0, 1.0))
    return img, steering


# ─────────────────────────────────────────────
# Quick Self-Test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys

    print("[INFO] Running data_augmentation self-test with a synthetic image …")

    # Create a random 160×320×3 BGR image (typical simulator resolution)
    dummy_img      = (np.random.rand(160, 320, 3) * 255).astype(np.uint8)
    dummy_steering = 0.1

    augmented_img, augmented_steering = augment_image(dummy_img, dummy_steering)

    assert augmented_img.shape == dummy_img.shape, \
        f"Shape mismatch: {augmented_img.shape} != {dummy_img.shape}"
    assert -1.0 <= augmented_steering <= 1.0, \
        f"Steering out of range: {augmented_steering}"

    print(f"[PASS] Augmented image shape  : {augmented_img.shape}")
    print(f"[PASS] Original steering      : {dummy_steering:.4f}")
    print(f"[PASS] Augmented steering     : {augmented_steering:.4f}")
    print("[PASS] data_augmentation.py — all checks passed.")
