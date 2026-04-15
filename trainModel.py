import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from dataLoader import load_data, balance_data
from batchGenerator import batch_generator
from model import build_model


# --- Config ---
CSV_PATH = 'driving_log.csv'
MODEL_PATH = 'model.h5'
LOSS_PLOT_PATH = 'loss_curve.png'
BATCH_SIZE = 32
EPOCHS = 10
VAL_SPLIT = 0.2


def main():
    # Requirement 6.1: load and split dataset 80/20
    print("Loading data...")
    image_paths, steering_angles = load_data(CSV_PATH)
    image_paths, steering_angles = balance_data(image_paths, steering_angles)
    print(f"Balanced dataset size: {len(image_paths)}")

    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, steering_angles,
        test_size=VAL_SPLIT,
        random_state=42
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    # Batch generators
    train_gen = batch_generator(X_train, y_train, BATCH_SIZE, is_training=True)
    val_gen = batch_generator(X_val,   y_val,   BATCH_SIZE, is_training=False)

    steps_per_epoch = math.ceil(len(X_train) / BATCH_SIZE)
    validation_steps = math.ceil(len(X_val) / BATCH_SIZE)

    # Build model
    model = build_model()
    model.summary()

    # Requirement 6.1, 6.4: train and capture history for loss curves
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        verbose=1,
    )

    # Requirement 6.2: plot and save training vs validation loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'],     label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH)
    print(f"Loss curve saved to '{LOSS_PLOT_PATH}'")

    # Requirement 6.3: save trained model
    model.save(MODEL_PATH)
    print(f"Model saved to '{MODEL_PATH}'")


if __name__ == '__main__':
    main()
