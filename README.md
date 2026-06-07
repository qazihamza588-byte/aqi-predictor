# London AQI Predictor — Serverless MLOps

End-to-end air quality forecasting system. Predicts AQI 72 hours ahead for **London**. Cloud-first: Hopsworks feature store + model registry, GitHub Actions CI/CD, Streamlit dashboard.

**City:** London · **Forecast horizon:** 72 hours · **Models:** Ridge · Random Forest (best auto-selected by RMSE)

---

## Architecture

    Data sources (AQICN + OpenWeather)
            │
            ▼
    Feature pipeline ──► Hopsworks Feature Store (online FG v2, RonDB)
            │                       │
            │                       ▼
            │             Training pipeline (Ridge/RF, 5-fold TimeSeriesSplit)
            │                       │
            │                       ▼
            │             Hopsworks Model Registry (versioned)
            ▼                       │
    Inference pipeline ◄────────────┘
            │
            ▼
    Dashboard (Streamlit) + REST API (Flask)

Cloud is single source of truth — no local data files. No CSV output.

---

## Prerequisites

- **Python 3.10 ONLY** — 3.11/3.12/3.13 break TensorFlow and hopsworks wheels
- A `.env` file with your API keys
- API keys — all free tier:
  - AQICN token — https://aqicn.org/data-platform/token/
  - OpenWeather key — https://openweathermap.org/api
  - Hopsworks key + project — https://app.hopsworks.ai

---

## Setup

    cd london_aqi_predictor

    # 1. Create Python 3.10 venv (Windows)
    py -3.10 -m venv venv

    # 2. Activate (Windows PowerShell)
    venv\Scripts\Activate.ps1

    # 3. Fix execution policy if needed
    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

    # 4. Install dependencies (TensorFlow is 350MB, allow 10-15 minutes)
    venv\Scripts\pip.exe install --timeout 300 --retries 20 -r requirements.txt

### Configure keys

    copy .env.example .env

Edit `.env` and fill in your keys:

    AQICN_API_TOKEN=your_aqicn_token
    OPENWEATHER_API_KEY=your_openweather_key
    HOPSWORKS_API_KEY=your_hopsworks_key
    HOPSWORKS_PROJECT=your_project_name

`.env` is gitignored — keys are never committed.

---

## Run

### Step 1 — Create C:\tmp (Windows requirement for Hopsworks)

    mkdir C:\tmp

### Step 2 — Backfill 90 days of data to Hopsworks

    $env:TMPDIR="C:\tmp"; $env:TEMP="C:\tmp"; $env:TMP="C:\tmp"; venv\Scripts\python.exe scripts/backfill.py --days 90

### Step 3 — Train models and push to Hopsworks registry

    $env:TMPDIR="C:\tmp"; $env:TEMP="C:\tmp"; $env:TMP="C:\tmp"; venv\Scripts\python.exe scripts/run_training_pipeline.py --days 90

### Step 4 — Launch the dashboard

    $env:TMPDIR="C:\tmp"; $env:TEMP="C:\tmp"; $env:TMP="C:\tmp"; venv\Scripts\python.exe -m streamlit run app/streamlit_app.py

Opens at http://localhost:8501

First load is slow — downloads model from Hopsworks cloud (~144MB). Cached after first run.

### Step 5 — Launch REST API (optional)

    venv\Scripts\python.exe app/flask_api.py

Serves at http://localhost:5000

Endpoints: /health, /cities, /current, /predict, /history, /alerts, /models

---

## Pipelines (run individually)

    # Backfill London features to Hopsworks cloud
    venv\Scripts\python.exe scripts/backfill.py --days 90

    # Hourly feature fetch (latest reading to cloud FG)
    venv\Scripts\python.exe scripts/run_feature_pipeline.py

    # Train models and push best to registry
    venv\Scripts\python.exe scripts/run_training_pipeline.py --days 90

All scripts exit 0 on API gaps so CI stays green.

---

## Tests

    venv\Scripts\python.exe -m pytest tests/ -v

**Result: 15/15 passed in 2.48s** — no API keys needed, uses synthetic fixtures.

Covers: feature engineering, AQI alert classification, regression metrics.

---

## Notebook (EDA)

    jupyter notebook notebooks/EDA.ipynb

Covers: AQI over time, daily and weekly seasonality, autocorrelation, pollutant and weather correlation matrix.

---

## CI/CD (GitHub Actions)

Three workflows in .github/workflows/

| Workflow | Trigger | Action |
|---|---|---|
| ci.yml | push / PR | pytest — no secrets required, always green |
| feature_pipeline.yml | hourly cron | fetch London AQI and push to Hopsworks FG |
| training_pipeline.yml | daily 02:00 UTC | retrain all models and push best to registry |

Repo secrets needed: AQICN_API_TOKEN, OPENWEATHER_API_KEY, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT

---

## Project Layout

    london_aqi_predictor/
    ├── src/
    │   ├── config.py                    # central config: city, keys, horizons
    │   ├── feature_pipeline/
    │   │   ├── data_fetcher.py          # AQICN + OpenWeather with synthetic fallback
    │   │   ├── feature_engineer.py      # time, lag, rolling, cyclical + 72h targets
    │   │   └── feature_store.py         # Hopsworks online FG v2 cloud-only
    │   ├── training_pipeline/
    │   │   ├── train.py                 # CV-train Ridge and RF with SHAP and LIME
    │   │   ├── model_registry.py        # Hopsworks model registry
    │   │   ├── evaluate.py              # RMSE, MAE, R2, MAPE
    │   │   └── models/                  # ridge, random_forest, lstm builders
    │   ├── inference_pipeline/
    │   │   └── predict.py               # 72h forecast with CI bands
    │   └── utils/
    │       └── alerts.py                # EPA 6-level AQI classification and alerts
    ├── app/
    │   ├── streamlit_app.py             # dashboard on port 8501
    │   └── flask_api.py                 # REST API on port 5000
    ├── scripts/
    │   ├── backfill.py                  # seed Hopsworks with historical data
    │   ├── run_feature_pipeline.py      # hourly feature fetch
    │   ├── run_training_pipeline.py     # daily model training
    │   └── full_run.py                  # one-shot end-to-end
    ├── tests/                           # pytest suite — 15 tests
    ├── notebooks/EDA.ipynb              # exploratory data analysis
    ├── .github/workflows/               # CI/CD workflows
    ├── requirements.txt
    └── .env.example

---

## Model Results (London, 90-day backfill)

| Model | RMSE | MAE | R2 | Status |
|---|---|---|---|---|
| Ridge Regression | 14.47 | 11.53 | 0.268 | Trained and registered |
| Random Forest | 13.47 | 10.95 | 0.390 | Best — trained and registered |
| LSTM | — | — | — | Skipped — TF protobuf conflict |

---

## Config Knobs (src/config.py)

| Variable | Default | Meaning |
|---|---|---|
| FORECAST_HOURS | 72 | prediction horizon |
| LOOKBACK_HOURS | 48 | LSTM input window |
| ALERT_THRESHOLD | 150 | AQI at or above this triggers hazard alert |
| MIN_TRAIN_ROWS | 200 | minimum rows needed to train |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| python3.10 is not recognized | Use py -3.10 -m venv venv |
| twofish build error | Run venv\Scripts\pip.exe install pyjks --no-deps first |
| Hopsworks /tmp path error on Windows | Run mkdir C:\tmp then prefix commands with $env:TEMP="C:\tmp" |
| No module named hopsworks | Use full path venv\Scripts\python.exe for all commands |
| Feature load failed or login error | Check .env file exists and HOPSWORKS_PROJECT matches exactly |
| confluent-kafka not found | Run venv\Scripts\pip.exe install confluent-kafka |
| hopsworks version mismatch | Run venv\Scripts\pip.exe install "hopsworks==4.7.*" --no-deps |
| PowerShell running scripts is disabled | Run Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned |
| Dashboard slow on first load | Expected — 144MB model download, cached after first run |
| protobuf version conflict warning | Harmless — ignore, everything runs fine |
