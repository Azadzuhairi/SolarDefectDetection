# SolarScan — Setup & Run Guide
University of Malaya FYP

## Folder structure expected
Place your Roboflow export so it looks like this:

```
your_project/
├── app.py
├── train.py
├── preprocessing.py
├── evaluate.py
├── predict.py
└── dataset2/
    ├── data.yaml          ← copy the fixed data.yaml here
    ├── train/
    │   ├── images/        ← training images
    │   └── labels/        ← .txt annotation files
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

The Roboflow v2 export already has train/, valid/, test/ folders.
Just rename the export folder to `dataset2/` and you are done.

---

## Step-by-step run order

### Step 1 — Install dependencies
```bash
pip install ultralytics torch opencv-python matplotlib seaborn
pip install streamlit plotly folium streamlit-folium pandas
```

### Step 2 — Preprocess images (run ONCE only)
```bash
python preprocessing.py
```
This converts all thermal images to INFERNO colormap.
Do NOT run it a second time.

### Step 3 — Train
```bash
python train.py
```
Expected time:
- CPU: ~4–8 hours for 150 epochs
- GPU (Colab T4): ~25–40 minutes

### Step 4 — Evaluate
```bash
python evaluate.py
```
Check mAP@0.5 in the output.

### Step 5 — Test inference
```bash
python predict.py
# or on a single image:
python predict.py --source dataset2/test/images/your_image.jpg
```

### Step 6 — Launch the web app
```bash
streamlit run app.py
```

---

## Expected results with 129 images

| Metric | Expected range | Notes |
|---|---|---|
| mAP@0.5 | 0.45 – 0.65 | Acceptable for FYP |
| Best class | Hotspot | Most samples |
| Weakest class | Junction_box | Fewest samples |
| Overfitting risk | Medium-High | Small dataset |

If mAP < 0.30, check:
1. Annotations in Roboflow — are bounding boxes tight?
2. Class distribution — are some classes missing labels entirely?
3. Preprocessing — was it run twice accidentally?

---

## Classes (7)

| ID | Name | Severity | Action |
|---|---|---|---|
| 0 | Hotspot | Investigate | Confirm defect type |
| 1 | Junction_box | High | Inspect wiring & seals |
| 2 | Mild | Low | Monitor |
| 3 | Moderate | Medium | Schedule inspection |
| 4 | Moderate-Critical | Critical | Immediate — diode failure |
| 5 | Moderate-High | High | Urgent inspection |
| 6 | healthy_panel | None | No action |

---

## Key fix summary (what changed from original scripts)

1. **data.yaml** — `path: dataset2`, paths corrected to `train/images`, `valid/images`, `test/images`
2. **train.py** — DATA_CONFIG now points to `dataset2/data.yaml`; 7 CLASS_NAMES including Junction_box; epochs bumped to 150; stronger augmentation for small dataset
3. **preprocessing.py** — IMAGE_DIRS corrected to `dataset2/train/images` etc.
4. **evaluate.py** — CLASS_NAMES updated to 7; DATA_CONFIG corrected
5. **predict.py** — CLASS_NAMES updated to 7 including Junction_box with its own color/action
