# SolarScan ☀️ — PV Panel Defect Detection App

A Streamlit web app for detecting and monitoring defects in solar PV panels
using YOLOv8 on both IR (infrared) and RGB images.

---

## 📁 Project Structure

```
solarscan/
├── app.py                  ← main entry point
├── requirements.txt
├── assets/
│   └── style.css           ← global dark theme styles
├── pages/
│   ├── detect.py           ← single image upload & detection
│   ├── dashboard.py        ← stats overview & charts
│   ├── batch.py            ← batch upload multiple images
│   ├── map_view.py         ← solar farm panel grid map
│   ├── compare.py          ← before vs after comparison
│   └── alerts.py           ← notification & alert system
└── utils/
    └── model.py            ← YOLOv8 model loading & inference
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Add your trained model weights
```
Place your trained weights at:
runs/pv_defect/exp/weights/best.pt
```
If no weights are found, the app runs in demo mode.

### 3. Run the app
```bash
streamlit run app.py
```

The app opens at: **http://localhost:8501**

---

## 🔗 Connect Your YOLOv8 Model

Once you have trained your model (using `train_pv_yolov8.py`), copy the weights:
```bash
# After training completes, your weights are at:
runs/pv_defect/exp/weights/best.pt

# The app will automatically detect and use them
```

---

## 📱 App Pages

| Page       | What it does                                      |
|------------|---------------------------------------------------|
| Detect     | Upload single IR or RGB image, run detection      |
| Dashboard  | View stats, defect types, weekly scan charts      |
| Batch      | Upload & process up to 50 images at once          |
| Map View   | Color-coded 8×8 grid of your solar farm panels    |
| Compare    | Side-by-side before & after comparison            |
| Alerts     | Notification feed + alert settings                |

---

## 🏷️ Defect Classes

| Class              | Severity |
|--------------------|----------|
| hotspot_minor      | LOW      |
| hotspot_moderate   | MEDIUM   |
| hotspot_severe     | HIGH     |
| bypass_diode_fault | HIGH     |
| soiling_ir         | LOW      |
| crack_minor        | LOW      |
| crack_severe       | HIGH     |
| delamination       | MEDIUM   |
| discoloration      | MEDIUM   |
| broken_cell        | HIGH     |
| soiling_rgb        | LOW      |
| snail_trail        | MEDIUM   |
| pid_defect         | HIGH     |
