"""
train.py
CVI620 Final Project — Winter 2026
Self-Driving Car Simulation using CNN

Builds the Nvidia End-to-End CNN (Figure 7 from the project spec), trains it
on the collected simulator data, plots the loss curves, and saves the final
model to models/model.h5.

Architecture (Figure 7):
  Input       : 66×200×3  (YUV, normalised)
  Conv2D      : 24  filters, 5×5, stride 2, ELU
  Conv2D      : 36  filters, 5×5, stride 2, ELU
  Conv2D      : 48  filters, 5×5, stride 2, ELU
  Conv2D      : 64  filters, 3×3, stride 1, ELU
  Conv2D      : 64  filters, 3×3, stride 1, ELU
  Flatten
  Dense(1164) : ELU
  Dense(100)  : ELU
  Dense(50)   : ELU
  Dense(10)   : ELU
  Dense(1)    : Linear  → steering angle
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, Flatten, Dense, Dropout, Lambda, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from data_preprocessing import (
    load_data, balance_dataset, split_data,
    preprocess_image, SAMPLES_PER_BIN, IMG_HEIGHT, IMG_WIDTH
)
from data_augmentation import augment_image


# ─────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────
BATCH_SIZE   = 100
EPOCHS       = 40
LEARNING_RATE = 1e-4
STEPS_PER_EPOCH_FACTOR = 0.8   # fraction of train set used per epoch
VAL_STEPS_FACTOR       = 1.0   # use whole val set each epoch

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "model.h5")


# ─────────────────────────────────────────────
# 1. Batch Generator
# ─────────────────────────────────────────────
def batch_generator(image_paths: np.ndarray, steering: np.ndarray,
                    batch_size: int = BATCH_SIZE, is_training: bool = True):
    """
    Infinite generator that yields (images, steerings) batches.

    During training: augmentation is applied on-the-fly.
    During validation: only preprocessing is applied (no augmentation).

    Parameters
    ----------
    image_paths : array of file paths
    steering    : array of steering angles
    batch_size  : number of samples per batch
    is_training : if True, apply random augmentation
    """
    num_samples = len(image_paths)
    indices     = np.arange(num_samples)

    while True:
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            batch_idx   = indices[start:start + batch_size]
            batch_imgs  = []
            batch_steer = []

            for i in batch_idx:
                img = cv2.imread(image_paths[i])
                if img is None:
                    continue
                angle = float(steering[i])

                if is_training:
                    img, angle = augment_image(img, angle)

                img = preprocess_image(img)
                batch_imgs.append(img)
                batch_steer.append(angle)

            if len(batch_imgs) == 0:
                continue

            yield (np.array(batch_imgs, dtype=np.float32),
                   np.array(batch_steer, dtype=np.float32))


# ─────────────────────────────────────────────
# 2. Nvidia CNN Model (Figure 7)
# ─────────────────────────────────────────────
def build_nvidia_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)) -> Sequential:
    """
    Construct the Nvidia End-to-End Self-Driving CNN.

    The model takes a normalised YUV image of shape (66, 200, 3) and outputs
    a single scalar representing the predicted steering angle.

    Reference: Bojarski et al., "End to End Learning for Self-Driving Cars" (2016)
    """
    model = Sequential(name="NvidiaSelfDrivingCNN")

    # ── Convolutional feature extractor ────────────────────────────────────
    # Three strided convolutions (stride=2) to downsample aggressively
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="elu",
                     input_shape=input_shape, name="conv1"))   # → 31×98×24
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="elu",
                     name="conv2"))                            # → 14×47×36
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="elu",
                     name="conv3"))                            # →  5×22×48

    # Two non-strided convolutions
    model.add(Conv2D(64, (3, 3), activation="elu", name="conv4"))  # → 3×20×64
    model.add(Conv2D(64, (3, 3), activation="elu", name="conv5"))  # → 1×18×64

    # ── Classifier head ─────────────────────────────────────────────────────
    model.add(Flatten())                                           # → 1164
    model.add(Dense(1164, activation="elu", name="fc1"))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation="elu",  name="fc2"))
    model.add(Dropout(0.3))
    model.add(Dense(50,  activation="elu",  name="fc3"))
    model.add(Dense(10,  activation="elu",  name="fc4"))
    model.add(Dense(1,                      name="output"))  # linear → steering

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"]
    )
    return model


# ─────────────────────────────────────────────
# 3. Plot Training History
# ─────────────────────────────────────────────
def plot_training_history(history, save_path: str = None):
    """
    Plot training and validation loss (MSE) and MAE curves side-by-side.
    Optionally save the figure to disk.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history["loss"],     label="Train Loss",      color="#2196F3")
    axes[0].plot(history.history["val_loss"], label="Validation Loss", color="#FF5722")
    axes[0].set_title("Mean Squared Error (Loss)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # MAE
    axes[1].plot(history.history["mae"],      label="Train MAE",       color="#4CAF50")
    axes[1].plot(history.history["val_mae"],  label="Validation MAE",  color="#FF9800")
    axes[1].set_title("Mean Absolute Error", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Nvidia CNN — Training Curves", fontsize=15, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Training curves saved to {save_path}")

    plt.show()


# ─────────────────────────────────────────────
# 4. Main Training Script
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  CVI620 Final Project — Nvidia CNN Training")
    print("=" * 60)

    # ── Load & preprocess split data ────────────────────────────
    image_paths, steering = load_data()
    image_paths, steering = balance_dataset(image_paths, steering)
    X_train, X_val, y_train, y_val = split_data(image_paths, steering)

    # ── Build model ─────────────────────────────────────────────
    model = build_nvidia_model()
    model.summary()

    # ── Steps-per-epoch calculation ─────────────────────────────
    steps_per_epoch  = max(1, int(len(X_train) * STEPS_PER_EPOCH_FACTOR // BATCH_SIZE))
    validation_steps = max(1, int(len(X_val)   * VAL_STEPS_FACTOR       // BATCH_SIZE))
    print(f"\n[INFO] steps_per_epoch  = {steps_per_epoch}")
    print(f"[INFO] validation_steps = {validation_steps}")

    # ── Callbacks ───────────────────────────────────────────────
    checkpoint = ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    # ── Train ───────────────────────────────────────────────────
    print(f"\n[INFO] Starting training for up to {EPOCHS} epochs …")
    history = model.fit(
        batch_generator(X_train, y_train, BATCH_SIZE, is_training=True),
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=batch_generator(X_val, y_val, BATCH_SIZE, is_training=False),
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    # ── Final save ──────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\n[INFO] Model saved to: {MODEL_PATH}")

    # ── Plot curves ─────────────────────────────────────────────
    plot_save = os.path.join(MODELS_DIR, "training_curves.png")
    plot_training_history(history, save_path=plot_save)

    print("\n[DONE] Training complete!")
    print(f"       Final val_loss : {min(history.history['val_loss']):.6f}")
    print(f"       Final val_mae  : {min(history.history['val_mae']):.6f}")


if __name__ == "__main__":
    main()
