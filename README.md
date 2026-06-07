# London AQI Predictor

End-to-end serverless AQI forecasting system for London.

**City:** London  
**Forecast horizon:** 72 hours (t+1h … t+72h, direct multi-output — not autoregressive)  
**Models:** Ridge Regression · Random Forest (best auto-selected by RMSE)  
**Feature Store & Model Registry:** Hopsworks Cloud  
**Dashboard:** Streamlit  
**API:** Flask REST  
**CI/CD:** GitHub Actions  

## Quick Start

```bash
py -3.10 -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/backfill.py --days 90
python scripts/run_training_pipeline.py --days 90
python -m streamlit run app/streamlit_app.py
```

## Architecture

AQICN + OpenWeather → Feature Pipeline → Hopsworks Feature Store → Training Pipeline → Hopsworks Model Registry → Inference → Streamlit Dashboard + Flask API
