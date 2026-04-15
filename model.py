from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense


def build_model():
    """
    Build and compile the Nvidia end-to-end CNN for steering angle regression.

    Architecture (Requirements 5.1, 5.2):
        Input  : (66, 200, 3)  — preprocessed YUV image
        Conv2D : 24 filters, 5x5, stride 2, elu
        Conv2D : 36 filters, 5x5, stride 2, elu
        Conv2D : 48 filters, 5x5, stride 2, elu
        Conv2D : 64 filters, 3x3, elu
        Conv2D : 64 filters, 3x3, elu
        Flatten
        Dense  : 100, elu
        Dense  : 50,  elu
        Dense  : 10,  elu
        Dense  : 1    (steering angle output)

    Compiled with loss='mse', optimizer='adam', metrics=['mae'] (Requirement 5.3).

    Returns:
        Compiled keras Sequential model.
    """
    model = Sequential([
        Input(shape=(66, 200, 3)),

        # --- Convolutional feature extractor ---
        Conv2D(24, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(64, (3, 3), activation='elu'),
        Conv2D(64, (3, 3), activation='elu'),

        # --- Fully connected regression head ---
        Flatten(),
        Dense(100, activation='elu'),
        Dense(50,  activation='elu'),
        Dense(10,  activation='elu'),
        Dense(1),   # single steering angle output
    ])

    model.compile(loss='mse', optimizer='adam')
    return model
