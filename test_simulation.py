"""
test_simulation.py
CVI620 Final Project — Winter 2026
Self-Driving Car Simulation using CNN

This script acts as the autonomous-driving server that connects to the Udacity
simulator running in AUTONOMOUS MODE.

How it works:
  1. The simulator connects via Socket.IO on port 4567 (WebSocket).
  2. On each "telemetry" event the simulator sends:
       - image   : base64-encoded PNG from the front camera
       - speed   : current vehicle speed
  3. This server:
       a. Decodes the image
       b. Runs the full preprocessing pipeline
       c. Feeds the image through the trained Nvidia CNN
       d. Sends back a "steer_angle" + "throttle" control command

Run this script BEFORE switching the simulator to Autonomous Mode:
    python test_simulation.py

Then, in the simulator:
    1. Choose the same track used during training
    2. Click "Autonomous Mode"

Requirements: flask, python-socketio==4.6.1, python-engineio==3.14.2, eventlet
"""

import os
import base64
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

import eventlet
import eventlet.wsgi
import socketio
from flask import Flask

import tensorflow as tf

from data_preprocessing import preprocess_image


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "models", "model.h5")
PORT         = 4567
SPEED_LIMIT  = 15.0   # mph — throttle is cut if speed exceeds this
MAX_THROTTLE = 0.3
MIN_THROTTLE = 0.1


# ─────────────────────────────────────────────
# Load trained model
# ─────────────────────────────────────────────
def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"[ERROR] Trained model not found at: {model_path}\n"
            "  → Run train.py first to generate models/model.h5"
        )
    print(f"[INFO] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("[INFO] Model loaded successfully.")
    return model


# ─────────────────────────────────────────────
# Socket.IO setup
# ─────────────────────────────────────────────
sio  = socketio.Server()
app  = Flask(__name__)
app  = socketio.Middleware(sio, app)

model = None   # populated in main()


def send_control(steering_angle: float, throttle: float):
    """Emit a control command to the simulator."""
    sio.emit("steer", {
        "steering_angle": str(steering_angle),
        "throttle":       str(throttle)
    })


# ─────────────────────────────────────────────
# Socket.IO Event Handlers
# ─────────────────────────────────────────────
@sio.on("connect")
def on_connect(sid, environ):
    print(f"[EVENT] Simulator connected  (sid={sid})")
    send_control(0, 0)


@sio.on("disconnect")
def on_disconnect(sid):
    print(f"[EVENT] Simulator disconnected (sid={sid})")


@sio.on("telemetry")
def on_telemetry(sid, data):
    """
    Receive camera image + speed from the simulator, predict steering,
    and send back a control command.
    """
    if data is None:
        send_control(0, 0)
        return

    # ── 1. Decode base64 image ──────────────────────────────────
    img_base64 = data["image"]
    img_bytes  = base64.b64decode(img_base64)
    pil_img    = Image.open(BytesIO(img_bytes))
    # Convert PIL (RGB) → OpenCV (BGR)
    img_bgr    = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ── 2. Preprocess ───────────────────────────────────────────
    img_processed = preprocess_image(img_bgr)
    img_input     = np.expand_dims(img_processed, axis=0)  # (1, 66, 200, 3)

    # ── 3. Predict steering angle ───────────────────────────────
    steering_angle = float(model.predict(img_input, verbose=0)[0][0])
    steering_angle = float(np.clip(steering_angle, -1.0, 1.0))

    # ── 4. Compute throttle (speed-based governing) ─────────────
    current_speed = float(data.get("speed", 0))
    if current_speed >= SPEED_LIMIT:
        throttle = MIN_THROTTLE
    else:
        # Scale throttle: faster approach → less throttle; slow → more
        throttle = MAX_THROTTLE - (abs(steering_angle) * 0.15)
        throttle = float(np.clip(throttle, MIN_THROTTLE, MAX_THROTTLE))

    print(f"[CTRL] steering={steering_angle:+.4f}  throttle={throttle:.3f}  "
          f"speed={current_speed:.1f} mph")

    # ── 5. Send control command ─────────────────────────────────
    send_control(steering_angle, throttle)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  CVI620 — Self-Driving Car Inference Server")
    print("=" * 60)

    # Load the trained model
    try:
        model = load_model()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print(f"\n[INFO] Server listening on port {PORT} …")
    print("[INFO] Switch the simulator to AUTONOMOUS MODE now.\n")

    listener = eventlet.listen(("0.0.0.0", PORT))
    eventlet.wsgi.server(listener, app)
