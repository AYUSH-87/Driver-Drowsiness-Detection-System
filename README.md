# 😴 Driver Drowsiness Detection

A real-time driver drowsiness detection system that uses deep learning and computer vision to monitor a driver's eyes and mouth via webcam — and triggers an audio alert when drowsiness or yawning is detected.

---

## 🚗 How It Works

The system uses two CNN models trained separately — one for eye state classification and one for mouth state classification. At runtime, OpenCV's Haar Cascade detectors locate the face, eyes, and mouth in each webcam frame. The cropped regions are passed to the respective models for prediction, and if closed eyes or an open mouth persist beyond a threshold number of frames, an audio beep alert fires.

**Detection logic:**
- 👁️ **Drowsy** — Eyes detected as closed for more than 5 consecutive frames
- 🥱 **Yawning** — Mouth detected as open for more than 10 consecutive frames
- ✅ **Alert** — Neither condition triggered

---

## 📁 Project Structure

```
driver-drowsiness-detection/
│
├── Driver_Drowsiness_Detection.ipynb   # Model training notebook (Google Colab)
├── main.py                             # Real-time detection script
├── eye_model.keras                     # Trained eye state CNN model
├── mouth_model.keras                   # Trained mouth state CNN model
└── README.md
```

> ⚠️ The `.keras` model files are not included in this repo due to file size. See the [Training](#-model-training) section to generate them yourself.

---

## 🧠 Model Architecture

Both the eye and mouth models share the same CNN architecture:

```
Input (64x64 grayscale)
  → Conv2D(32, 3x3, ReLU) → MaxPooling
  → Conv2D(64, 3x3, ReLU) → MaxPooling
  → Flatten
  → Dense(128, ReLU)
  → Dense(1, Sigmoid)     ← Binary output
```

- **Eye model** — classifies `eyes_open` vs `eyes_closed`
- **Mouth model** — classifies `mouth_open` vs `mouth_closed`
- **Optimizer:** Adam | **Loss:** Binary Crossentropy
- **Input size:** 64×64 pixels, grayscale

---

## 📊 Dataset

The dataset is too large to include in this repository. You can download it from Google Drive:

📥 **[Download Dataset from Google Drive](https://drive.google.com/drive/folders/1q9lo-5M8Fbij32LY_gtV5Aijjnn8Gaji?usp=sharing)**

After downloading, organize it as follows:

```
final_dataset/
├── eyes/
│   ├── train/   (1814 images)
│   ├── val/     (388 images)
│   └── test/    (390 images)
└── mouth/
    ├── train/
    ├── val/
    └── test/
```

Both datasets use binary class folders — e.g., `eyes_open/` and `eyes_closed/`.

---

## ⚙️ Requirements

- Python 3.8+
- Windows OS *(for `winsound` audio alerts)*
- Webcam

Install dependencies:

```bash
pip install opencv-python tensorflow numpy
```

---

## 🏋️ Model Training

Open `Driver_Drowsiness_Detection.ipynb` in **Google Colab**, mount your Google Drive, and set the dataset path:

```python
BASE_PATH = "/content/drive/MyDrive/final_dataset"
```

Run all cells to train both models. At the end, the notebook will download:
- `eye_model.keras`
- `mouth_model.keras`

Place both files in the **same directory as `main.py`** before running the detection script.

---

## ▶️ Running the Detection

```bash
python main.py
```

- Your webcam will open with a live status overlay
- The status bar at the top will show **ALERT** (green), **DROWSY** (red), or **YAWNING** (red)
- An audio beep fires when drowsiness or yawning is detected (with a 2-second cooldown)
- Press **`Q`** to quit

---

## 🔍 Detection Parameters

You can tune these values in `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `EYE_THRESH` | `5` frames | Frames of closed eyes before alert |
| `MOUTH_THRESH` | `10` frames | Frames of open mouth before alert |
| `ALERT_COOLDOWN` | `2` seconds | Minimum time between repeated alerts |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| TensorFlow / Keras | CNN model training & inference |
| OpenCV | Webcam capture, Haar Cascade detection, UI overlay |
| NumPy | Image preprocessing |
| winsound | Audio alert beeps |
| Google Colab | Model training environment |

---

## 📌 Notes

- The system currently supports single-face detection.
- `winsound` is Windows-only. For Linux/macOS, replace it with `playsound` or `pygame`.
- Lighting conditions can affect Haar Cascade detection accuracy.

---

## 🙌 Acknowledgements

- [OpenCV Haar Cascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)
- [TensorFlow / Keras](https://www.tensorflow.org/)
