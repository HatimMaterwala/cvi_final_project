import cv2
import numpy as np

from dataAugmentation import random_augment
from dataPreprocessing import preprocess


def batch_generator(image_paths, steering_angles, batch_size, is_training):
    """
    Infinite generator that yields (images, steerings) batches.

    Requirements:
      4.1 - yields batches of configurable size
      4.2 - training batches apply random_augment then preprocess
      4.3 - validation batches apply preprocess only (no augmentation)
      2.4 - augmentation is applied ONLY when is_training=True

    Args:
        image_paths    : array-like of file path strings
        steering_angles: array-like of float steering values
        batch_size     : int, number of samples per batch
        is_training    : bool, True → augment + preprocess, False → preprocess only

    Yields:
        (np.ndarray of shape (batch_size, 66, 200, 3),
         np.ndarray of shape (batch_size,))
    """
    num_samples = len(image_paths)
    indices = np.arange(num_samples)

    while True:
        # Shuffle each epoch during training for better generalisation
        if is_training:
            np.random.shuffle(indices)

        for start in range(0, num_samples, batch_size):
            batch_idx = indices[start:start + batch_size]

            batch_images = []
            batch_steerings = []

            for i in batch_idx:
                path = image_paths[i]
                angle = steering_angles[i]

                # Load image; skip with warning on read failure
                img = cv2.imread(path)
                if img is None:
                    print(
                        f"[WARNING] Could not load image: {path} — skipping.")
                    continue

                # cv2 reads BGR; convert to RGB for augmentation functions
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if is_training:
                    img, angle = random_augment(img, angle)  # Req 4.2, 2.4

                img = preprocess(img)  # Req 4.1, 4.3

                batch_images.append(img)
                batch_steerings.append(angle)

            if batch_images:
                yield np.array(batch_images, dtype=np.float32), \
                    np.array(batch_steerings, dtype=np.float32)
