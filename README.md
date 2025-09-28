# Neonatal Incubator Display Reader

End-to-end computer vision pipeline to detect neonatal incubator monitor regions, extract telemetry via OCR, validate accuracy, and expose results through a Streamlit dashboard.

## Features

- Dataset exploration and preprocessing notebook (`notebooks/incubator_pipeline.ipynb`).
- Label Studio integration for bounding-box annotation in YOLO format.
- YOLOv8 detector training workflow with exportable weights (`models/incubator_yolov8n.pt`).
- EasyOCR-based text extraction with numeric sanitisation and confidence tracking.
- Streamlit dashboard for manual uploads and batch processing with CSV export.

## Project structure

```
incubator_pipeline/      # Reusable Python package for detection + OCR
notebooks/               # Jupyter notebooks (JSON format) documenting the workflow
streamlit_app/app.py     # Streamlit UI for live inference and logging
tests/                   # Lightweight regression tests for helper utilities
dataset/                 # Raw monitor images (needs manual annotation)
models/                  # Expected location for trained YOLO weights
```

## Quickstart

1. **Create a virtual environment** (Python 3.10+ recommended) and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. **Annotate the dataset** using Label Studio and export in YOLO or JSON format.
3. **Run the notebook** `notebooks/incubator_pipeline.ipynb` to:
   - Convert annotations to YOLO format
   - Train and evaluate the detector
   - Validate OCR performance
   - Export the best weights to `models/incubator_yolov8n.pt`
4. **Launch the dashboard**:

```powershell
cd streamlit_app
streamlit run app.py
```

Upload new photos or process a directory to log structured telemetry and export CSV reports.

## Testing

Run the minimal regression tests:

```powershell
pytest
```

## Next steps

- Expand annotations to cover new monitor layouts.
- Add automated data augmentation (Albumentations) for improved robustness.
- Integrate scheduled ingestion for continuous monitoring.
