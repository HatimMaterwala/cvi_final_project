# CVI620 — Self-Driving Car Simulation Using CNN

**Course:** CVI620 Computer Vision — Winter 2026  
**Instructor:** Ellie Azizi  
**Deadline:** April 14, 2026

---

## 📌 Project Overview

This project implements a **Convolutional Neural Network (CNN)** that learns to steer a simulated vehicle by predicting the correct steering angle from front-camera images. The implementation follows the **Nvidia End-to-End Self-Driving CNN** architecture ("End to End Learning for Self-Driving Cars", Bojarski et al., 2016).

The trained model is integrated with the **Udacity self-driving car simulator** via a Flask + Socket.IO server, enabling real-time autonomous driving.

---

## 🏗️ Project Structure

```
cvi_final_project/
├── data/                        # ← Simulator output goes here
│   ├── IMG/                     #     Camera images (center, left, right)
│   └── driving_log.csv          #     Driving telemetry CSV
├── models/                      # Auto-created during training
│   ├── model.h5                 #     Best trained model
│   └── training_curves.png      #     Loss / MAE plots
├── data_preprocessing.py        # Load, balance, split, preprocess images
├── data_augmentation.py         # Random augmentation (flip, brightness, etc.)
├── train.py                     # Build Nvidia CNN, train, save model
├── test_simulation.py           # Inference server for Autonomous Mode
├── requirements.txt             # Pinned dependencies
└── README.md                    # This file
```

---

## ⚙️ Environment Setup

> **Recommended Python version:** 3.9  

```bash
# 1. Clone the repository
git clone https://github.com/HatimMaterwala/cvi_final_project.git
cd cvi_final_project

# 2. Create a virtual environment
python3.9 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Important:** The `python-socketio==4.6.1` / `python-engineio==3.14.2` versions are critical — newer versions use a different Socket.IO protocol that is **incompatible** with the Udacity simulator client.

---

## 🎮 Data Collection (Udacity Simulator)

1. Download the Udacity self-driving car simulator for your OS.
2. Launch the simulator, choose **Training Mode**.
3. Click **Record** (top menu) and select the `data/` folder inside this project.
4. Drive the car around the track:
   - Drive **~5 laps forward** and **~5 laps in reverse** for balanced data.
   - Use the mouse for smooth, continuous steering inputs.
5. Stop recording — the simulator writes `data/IMG/` and `data/driving_log.csv`.

After collection you should have:
- `data/driving_log.csv` — telemetry log
- `data/IMG/` — thousands of `.jpg` images

---

## 🧠 Approach

### 1. Data Preprocessing (`data_preprocessing.py`)
- **Load:** Parse `driving_log.csv`, extract center-camera paths + steering angles.
- **Visualize:** Plot a histogram of steering angles to check distribution balance.
- **Balance:** Cap over-represented bins at 1 000 samples to prevent straight-driving bias.
- **Split:** 80 % training / 20 % validation.
- **Pipeline per image:**
  1. **Crop** rows 60–135 (remove sky and car hood — Figure 6)
  2. **BGR → YUV** colour space (matches Nvidia pre-processing)
  3. **Gaussian Blur** (3×3 kernel) to reduce noise
  4. **Resize** to 200×66 pixels (Nvidia model input)
  5. **Normalize** pixels to `[-1, 1]`

### 2. Data Augmentation (`data_augmentation.py`)
Applied **only to training data**, randomly (≈50 % probability per transform):
| Transform | Effect on Steering |
|---|---|
| Horizontal Flip | Negate angle |
| Brightness (HSV) | None |
| Pan (translate) | ± proportional offset |
| Zoom (crop+resize) | None |
| Rotation (±15°) | Small proportional offset |

### 3. Neural Network Architecture (`train.py`)
Follows the **Nvidia End-to-End CNN** (Figure 7):

```
Input:  66 × 200 × 3  (YUV, normalised)
──────────────────────────────────────────
Conv2D  24 × 5×5, stride 2, ELU   → 31×98×24
Conv2D  36 × 5×5, stride 2, ELU   → 14×47×36
Conv2D  48 × 5×5, stride 2, ELU   →  5×22×48
Conv2D  64 × 3×3, stride 1, ELU   →  3×20×64
Conv2D  64 × 3×3, stride 1, ELU   →  1×18×64
Flatten                            → 1 164
Dense(1164)  ELU + Dropout(0.3)
Dense(100)   ELU + Dropout(0.3)
Dense(50)    ELU
Dense(10)    ELU
Dense(1)     Linear  ← steering angle output
──────────────────────────────────────────
Loss: MSE  |  Optimizer: Adam (lr=1e-4)
```

### 4. Inference / Test Simulation (`test_simulation.py`)
- Starts a **Flask + Socket.IO** server on port `4567`.
- On each simulator frame: decode → preprocess → predict → send `steer` + `throttle`.
- Throttle is governed by current speed and magnitude of the steering angle.

---

## 🚀 Running Training

```bash
# Make sure your venv is active and data/ is populated
python train.py
```

Training runs for up to `40` epochs with `EarlyStopping` (patience=8).  
The best model is saved automatically to `models/model.h5`.  
A loss/MAE plot is saved to `models/training_curves.png` and displayed.

---

## 🏎️ Running the Simulation (Autonomous Mode)

```bash
# Step 1 — Start the inference server
python test_simulation.py

# Step 2 — Launch the simulator
#   - Choose the same track used for training
#   - Click "Autonomous Mode"
#   - Watch the car drive!
```

---

## 🧩 Major Challenges & Solutions

| Challenge | Solution |
|---|---|
| **Severely imbalanced data** (most angles ≈ 0) | Histogram-based bin capping (`balance_dataset`) |
| **Overfitting on straight road** | Dropout layers + heavy data augmentation |
| **Socket.IO version mismatch** | Pinned `python-socketio==4.6.1`, `python-engineio==3.14.2` |
| **Image path portability** | `resolve_path()` handles both absolute (Windows) and relative paths |
| **Throttle oscillation** | Speed-governed throttle with steering-angle damping |

---

## 📦 Dependencies

See [requirements.txt](requirements.txt) for the full pinned list.

Key packages:
- `tensorflow==2.13.0` — deep learning framework
- `opencv-python==4.8.0.76` — image processing
- `flask==2.3.3` + `python-socketio==4.6.1` — simulator communication
- `eventlet==0.33.3` — async WSGI server

---

## 📄 References

- Bojarski, M., et al. *End to End Learning for Self-Driving Cars*. NVIDIA, 2016.  
- Udacity Self-Driving Car Simulator — [github.com/udacity/self-driving-car-sim](https://github.com/udacity/self-driving-car-sim)
