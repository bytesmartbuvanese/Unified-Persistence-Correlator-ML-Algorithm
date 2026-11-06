# Student Withdrawal Early-Warning System

Predicts student withdrawal risk at course weeks 2, 4, and 6 using the Open University Learning Analytics Dataset (OULAD).

## Project Structure

```
dropout_risk/
├── engineered_data/                       # Engineered feature datasets
│   ├── engineered_weekly_features.csv
│   ├── engineered_weekly_features_with_affect.csv
│   └── engineered_student_course_features.csv
│
├── strict_early_warning_v2/               # ← TRAINED MODEL FILES
│   ├── week_2/
│   │   ├── component_models.joblib        # Engagement + affect + hazard models
│   │   ├── learned_correlator.joblib      # Calibrated stacking correlator
│   │   └── model_config.json             # Threshold & evaluation metrics
│   ├── week_4/
│   │   └── ...
│   ├── week_6/
│   │   └── ...
│   └── overall_summary.json              # Combined landmark results
│
├── api.py                                 # FastAPI inference service
├── strict_time_aware_early_warning_v2.py  # Model training & evaluation
├── feature_engineering.py                 # Feature engineering pipeline
├── improve_affect_estimation.py           # Synthetic affect proxy generation
├── preprocess_data.py                     # Raw data preprocessing
│
├── PROJECT_DOCUMENTATION.md               # Full technical documentation
├── pyproject.toml                         # Python dependencies
├── uv.lock                               # Lockfile
└── .python-version                        # Python 3.12+
```

## Model Files

The trained models live in `strict_early_warning_v2/`. Each week folder contains:

| File | What it is |
|------|------------|
| `component_models.joblib` | Dict with 3 fitted models: `engagement_model` (regressor), `affect_model` (multi-output regressor), `hazard_model` (HistGBM+RF voting classifier) |
| `learned_correlator.joblib` | Sigmoid-calibrated logistic stacking correlator |
| `model_config.json` | Validation-selected threshold + all evaluation metrics |

The API loads all three week bundles at startup and uses them for inference.

## Quick Start

### 1. Install dependencies

```powershell
uv sync
```

### 2. Start the API

```powershell
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Make a prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "STU001",
    "landmark_week": 2,
    "snapshot": {
      "week_number": 2,
      "cumulative_active_weeks": 2,
      "cumulative_active_days": 5,
      "cumulative_interactions": 15,
      "cumulative_clicks": 120,
      "days_since_registration": 14,
      "average_clicks_per_active_week": 60,
      "studied_credits": 60,
      "previous_week_clicks": 50,
      "weekly_total_clicks": 70,
      "click_change_from_previous_week": 20,
      "weekly_interaction_records": 8,
      "weekly_active_days": 3,
      "num_of_prev_attempts": 0,
      "hazard_low_active_days": 0
    }
  }'
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/`  | Service info |
| `GET`  | `/health` | Model loading status |
| `POST` | `/predict` | Withdrawal risk prediction |

## Retraining (optional)

To retrain models from the engineered data:

```powershell
uv run strict_time_aware_early_warning_v2.py
```

This reads `engineered_weekly_features_with_affect.csv`, trains all three landmark models, and overwrites the artifacts in `strict_early_warning_v2/`.

See [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for the full feature dictionary, evaluation results, and methodology.
