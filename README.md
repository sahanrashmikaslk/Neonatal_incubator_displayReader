# Neonatal Incubator Display Reader

End-to-end pipeline to detect incubator display regions (YOLOv8), run OCR, validate readings, and view results in a Streamlit UI. Dataset was collected on-site at National Hospital Galle, NICU (live incubator displays) and manually labeled in Label Studio for YOLO training.

Model training was carried out as part of the final-year research project _“Development of an Automated Condition Controlling and Monitoring System for an Infant Incubator.”_

This README matches the latest training notebook (`notebooks/incubator_lcd_reader_pipeline.ipynb`) and the app (`streamlit_app/app_opencv_webcam.py`).

---

## Layout (key parts)

- `notebooks/incubator_lcd_reader_pipeline.ipynb` — train/eval/export workflow; each run saves its own artifacts.
- `artifacts/` — data and outputs.
  - `yolo/images`, `yolo/labels` — YOLO-formatted dataset (matching basenames).
  - `runs/run_<timestamp>/` — splits, training logs, metrics, plots, config.
- `models/` — exported weights (best copy per run).
- `incubator_pipeline/` — core detection/OCR/validation utilities.
- `streamlit_app/` — Streamlit UI and its detector/OCR wrapper.

---

## Prerequisites

- Python 3.8+
- EasyOCR
- (Optional) CUDA GPU for faster training/inference.

Install deps:

```bash
pip install -r requirements.txt
pip install -r streamlit_app/requirements.txt
```

---

## Data preparation (YOLO)

1. Place images under `artifacts/yolo/images` and labels under `artifacts/yolo/labels`.
2. Filenames must match (`image.jpg` ↔ `image.txt`). If labels have prefixes, normalize them so the basename matches the image before splitting.
3. Dataset origin: collected on-site at National Hospital, Galle NICU, photographing live incubator displays (250+ images) for detector training.
4. Annotation: manually labeled with Label Studio to produce YOLO-format labels.

---

## Training & evaluation (notebook)

Open `notebooks/incubator_lcd_reader_pipeline.ipynb` and run:

1. **Run setup**: creates `artifacts/runs/run_<timestamp>/`.
2. **Split + data.yaml**: builds train/val/test splits from `artifacts/yolo`; writes `data.yaml` inside the run folder.
3. **Train YOLOv8**: early stopping (patience=10), max 100 epochs. Outputs under `runs/<run_tag>/train`.
4. **Eval**: saves `val_metrics.json` + plots (`results.png`, `f1_curve.png`, `pr_curve.png`, `confusion_matrix.png`) to `runs/<run_tag>/eval_plots/`.
5. **Export weights**: copies best.pt into the run and to `models/incubator_<run_tag>.pt`.
6. **Config snapshot**: `run_config.json` saved in the run folder.

Handy notebook cells:

- Split/data.yaml: `split_dataset(...)`, `write_data_yaml(...)`
- Train: `train_detector(data_yaml_path)`
- Eval + plots: `evaluate_detector(detector, data_yaml_path, RUN_ROOT)`
- OCR debug grid (optional): `visualize_ocr_grid(...)`

---

## Streamlit app

Run the UI:

```bash
cd streamlit_app
streamlit run app_opencv_webcam.py
```

What it does:

- Live webcam, upload, and batch tabs.
- Uses YOLO detector + OCR with validation.
- Sidebar: set weights path (use your exported `../models/incubator_<run_tag>.pt`), detection/OCR thresholds, ROI caching, FP16 toggle.

---

## OCR & validation

- OCR: EasyOCR in the core pipeline; Streamlit modes can use Tesseract.
- Validation (`incubator_pipeline/postprocessing.py` and notebook cells):
  - Range checks, integer/decimal enforcement.
  - Decimal fixes (e.g., 365 → 36.5).
  - Confidence filtering.
  - Optional temporal smoothing in the app for live video.

---

## Typical workflow

1. Prepare YOLO data under `artifacts/yolo/`.
2. Run the notebook: split → train → eval → export weights (check `artifacts/runs/<run_tag>/`).
3. Copy weights from `models/incubator_<run_tag>.pt`.
4. Launch Streamlit, point to the weights, and use live/upload/batch with validation.

---

## Model performance (latest run)

From `artifacts/runs/run_20251203_134740/val_metrics.json`:

- Precision: 0.994
- Recall: 0.992
- mAP@0.5: 0.993
- mAP@0.5:0.95: 0.745

Eval plots (same run) under `eval_plots/`:

- `results.png`
- `f1_curve.png`
- `pr_curve.png`
- `confusion_matrix.png`

---

## References

- YOLOv8 (Ultralytics)
- EasyOCR
- Streamlit
