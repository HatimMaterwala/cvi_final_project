import base64
import os
import time
import eventlet
import socketio
import numpy as np
import cv2
import tensorflow as tf
from io import BytesIO
from PIL import Image
from flask import Flask
from tensorflow.keras.models import load_model

# --- PERFORMANCE OPTIMIZATION ---
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

print("Setting Up Server for HIGH SPEED AUTONOMOUS MODE...")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

HOST = "0.0.0.0"
PORT = 4567

# ==============================================================================
# --- CONFIGURATION: OPTIMIZED FOR STABILITY ---
# ==============================================================================

# 1. Target Speed: Set to 3.0 for better stability.
TARGET_SPEED = 15.0

# 2. Steering Gain
STEERING_GAIN = 1.5

# 3. Smoothing: 0.2.
# Adding heavy smoothing back as a "shock absorber" so the wheel doesn't jerk.
EMA_ALPHA = 0.2

MODEL_CANDIDATES = ("model.h5", "model.keras")

sio = socketio.Server(cors_allowed_origins="*", logger=False, engineio_logger=False)
app = Flask(__name__)
model = None
last_steering = 0.0


from dataPreprocessing import preprocess as preProcessing

def send_control(sid, steering, throttle):
    sio.emit(
        "steer",
        data={
            "steering_angle": f"{steering}",
            "throttle": f"{throttle}",
        },
        room=sid
    )


def get_model_path():
    for candidate in MODEL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


@sio.on("telemetry")
def telemetry(sid, data):
    global last_steering

    if not data:
        print("No telemetry data received", flush=True)
        send_control(sid, last_steering, 0.15)
        return

    try:
        speed = float(data.get("speed", 0))

        if "image" not in data or not data.get("image"):
            print(f"Telemetry received without image. Keys: {list(data.keys())}", flush=True)
            throttle = 0.15 if speed < TARGET_SPEED else 0.0
            send_control(sid, last_steering, throttle)
            return

        image_bytes = base64.b64decode(data["image"])
        image = Image.open(BytesIO(image_bytes))
        image = np.asarray(image)

        prepped = preProcessing(image)
        prepped = np.expand_dims(prepped, axis=0)

        prediction = model(prepped, training=False)
        raw_steering = float(prediction[0][0])

        target_steering = raw_steering * STEERING_GAIN
        current_steering = (EMA_ALPHA * target_steering) + ((1 - EMA_ALPHA) * last_steering)
        last_steering = current_steering

        steering = max(min(current_steering, 1.0), -1.0)

        if speed > TARGET_SPEED:
            throttle = 0.0
        else:
            throttle = 0.30

        print(f"Spd: {speed:4.1f} | Str: {steering:6.3f} | Throt: {throttle:4.2f}", flush=True)
        send_control(sid, steering, throttle)

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        send_control(sid, last_steering, 0.0)

@sio.on("connect")
def connect(sid, environ):
    global last_steering
    print(f"CONNECTED: {sid} (High Speed Mode)", flush=True)
    last_steering = 0.0
    send_control(sid, 0, 0)


if __name__ == "__main__":
    model_path = get_model_path()
    if not model_path:
        print("Error: model file missing.")
        os._exit(1)

    model = load_model(model_path, compile=False)
    print(f"Model loaded from {model_path}")
    
    # Warmup
    dummy = np.zeros((1, 66, 200, 3), dtype=np.float32)
    model(dummy, training=False)

    app = socketio.Middleware(sio, app)
    print("-" * 50)
    print(f"MODE: STABILITY AUTONOMOUS")
    print(f"Target Speed: {TARGET_SPEED} | Gain: {STEERING_GAIN}")
    print("-" * 50)
    
    eventlet.wsgi.server(eventlet.listen(("", PORT)), app)