# London AQI Predictor — Serverless MLOps

End-to-end air quality forecasting system. Predicts AQI 72 hours ahead for **London**. Cloud-first: Hopsworks feature store + model registry, GitHub Actions CI/CD, Streamlit dashboard.

**City:** London · **Forecast horizon:** 72 hours (t+1h … t+72h, direct multi-output — not autoregressive) · **Models:** Ridge · Random Forest (best auto-selected by RMSE)

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

- **Python 3.10 ONLY** (REQUIRED — 3.11/3.12/3.13 break TensorFlow + hopsworks wheels)
- A `.env` file with your API keys (see Configure keys below)
- API keys (all free tier):
  - [AQICN token](https://aqicn.org/data-platform/token/)
  - [OpenWeather key](https://openweathermap.org/api)
  - [Hopsworks key + project](https://app.hopsworks.ai)

---

## Setup

```bash
cd london_aqi_predictor

# 1. Create Python 3.10 venv
py -3.10 -m venv venv

# 2. Activate (Windows PowerShell)
venv\Scripts\Activate.ps1

# 3. Install deps (TensorFlow ~350MB, allow time)
venv\Scripts\pip.exe install --timeout 300 --retries 20 -r requirements.txt
```

### Configure keys

```bash
copy .env.example .env
```

Edit `.env`:
AQICN_API_TOKEN=your_aqicn_token
OPENWEATHER_API_KEY=your_openweather_key
HOPSWORKS_API_KEY=your_hopsworks_key
HOPSWORKS_PROJECT=your_project_name

`.env` is gitignored — keys never committed.

---

## Run

### Full pipeline — one shot

```bash
mkdir C:\tmp
$env:TMPDIR="C:\tmp"; $env:TEMP="C:\tmp"; $env:TMP="C:\tmp"
venv\Scripts\python.exe scripts/backfill.py --days 90
venv\Scripts\python.exe scripts/run_training_pipeline.py --days 90
```

### Launch dashboard

```bash
$env:TMPDIR="C:\tmp"; $env:TEMP="C:\tmp"; $env:TMP="C:\tmp"
venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Opens at **http://localhost:8501**

### Launch REST API (optional)

```bash
venv\Scripts\python.exe app/flask_api.py
```

Serves at **http://localhost:5000**

---

## Pipelines (run individually)

```bash
# Backfill 90 days of London features to Hopsworks cloud
venv\Scripts\python.exe scripts/backfill.py --days 90

# Hourly feature fetch
venv\Scripts\python.exe scripts/run_feature_pipeline.py

# Train models and push to registry
venv\Scripts\python.exe scripts/run_training_pipeline.py --days 90
```

---

## Tests

```bash
venv\Scripts\python.exe -m pytest tests/ -v
```

**Result: 15/15 passed** — no API keys needed (uses synthetic fixtures).

---

## Notebook (EDA)

```bash
jupyter notebook notebooks/EDA.ipynb
```

Covers: AQI over time, daily/weekly seasonality, autocorrelation, pollutant correlation matrix.

---

## CI/CD (GitHub Actions)

| Workflow | Trigger | Action |
|---|---|---|
| `ci.yml` | push / PR | pytest (no secrets, always green) |
| `feature_pipeline.yml` | hourly cron | fetch London AQI to cloud FG |
| `training_pipeline.yml` | daily 02:00 UTC | retrain and push to registry |

**Repo secrets needed:** `AQICN_API_TOKEN`, `OPENWEATHER_API_KEY`, `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`

---

## Project Layout
london_aqi_predictor/
├── src/
│   ├── config.py
│   ├── feature_pipeline/
│   │   ├── data_fetcher.py
│   │   ├── feature_engineer.py
│   │   └── feature_store.py
│   ├── training_pipeline/
│   │   ├── train.py
│   │   ├── model_registry.py
│   │   ├── evaluate.py
│   │   └── models/
│   ├── inference_pipeline/
│   │   └── predict.py
│   └── utils/
│       └── alerts.py
├── app/
│   ├── streamlit_app.py
│   └── flask_api.py
├── scripts/
│   ├── backfill.py
│   ├── run_feature_pipeline.py
│   ├── run_training_pipeline.py
│   └── full_run.py
├── tests/
├── notebooks/EDA.ipynb
├── .github/workflows/
├── requirements.txt
└── .env.example

---

## Model Results (London, 90-day backfill)

| Model | RMSE | MAE | R² | Status |
|---|---|---|---|---|
| Ridge Regression | 14.47 | 11.53 | 0.268 | Registered |
| Random Forest | 13.47 | 10.95 | 0.390 | **Best — Registered** |
| LSTM | — | — | — | Skipped (TF/protobuf conflict) |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python3.10 is not recognized` | Use `py -3.10 -m venv venv` |
| `twofish` build error | Run `venv\Scripts\pip.exe install pyjks --no-deps` first |
| Hopsworks `/tmp` path error | Run `mkdir C:\tmp` then set `$env:TEMP="C:\tmp"` |
| `No module named 'hopsworks'` | Use full path `venv\Scripts\python.exe` |
| `Feature load failed` | Check `.env` file — verify HOPSWORKS_PROJECT name |
| `confluent-kafka` not found | Run `venv\Scripts\pip.exe install confluent-kafka` |
| `hopsworks` version mismatch | Run `venv\Scripts\pip.exe install "hopsworks==4.7.*" --no-deps` |
| PowerShell scripts disabled | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| Dashboard slow first load | Expected — 144MB model download, cached after first run |
| `protobuf` version conflict | Harmless warning — ignore |
