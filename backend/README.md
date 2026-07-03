# Pulse Diagnosis Prediction Backend

Production-ready Flask backend for the Final Year Research Project: **Designing an ML-Based Wearable Pulse Monitoring and Disease Prediction System with Android Application and Cloud Backend**.

## Architecture

```text
backend/
  app.py
  wsgi.py
  requirements.txt
  runtime.txt
  Procfile
  render.yaml
  .env.example
  README.md
  src/
    api/            Flask app factory and routes
    services/       Prediction business logic
    ml/             Model loading and artifact management
    models/         Domain model package
    database/       SQLite repository layer
    utils/          JSON and response helpers
    middleware/     Security headers
    config/         Environment-driven settings
    validators/     Request validation and feature ordering
    logging/        Structured logging
  trained_models/
    model.pkl
    scaler.pkl
    metadata.json
    metrics.json
  datasets/
  logs/
  tests/
  docs/
```

## Local Installation

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python app.py
```

## Required Model Artifacts

The backend loads these files at startup:

```text
trained_models/model.pkl
trained_models/scaler.pkl
trained_models/metadata.json
```

The legacy `models/` folder can remain, but production loading uses `trained_models/` by default.

## API

- `GET /`
- `GET /health`
- `GET /version`
- `POST /predict`
- `GET /metrics`
- `GET /docs`
- `GET /swagger`
- `GET /history`
- `DELETE /history`

Android aliases are preserved:

- `POST /api/v1/predict`
- `GET /api/v1/health`
- `GET /api/v1/history`
- `GET /api/v1/model-info`

## Prediction Input

The feature order is always fixed:

```text
mean, std, variance, min, max, energy
```

## Render Deployment

Use the repository root `PulseDiagnosis/render.yaml` if the Android project is the GitHub root. It points Render to `backend`.

Manual Render settings:

```text
Root Directory: backend
Build Command: pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
Start Command: gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
Health Check Path: /health
```

## Environment Variables

See `.env.example`. Important values include `PORT`, `MODEL_PATH`, `SCALER_PATH`, `DATABASE_PATH`, `CORS_ORIGINS`, and `RATE_LIMIT_PREDICT`.

## Android Integration

Use the deployed URL:

```java
private static final String PREDICTION_URL =
        "https://YOUR-RENDER-SERVICE.onrender.com/api/v1/predict";
```

The response still includes `prediction`, `confidence`, `risk_level`, and `prediction_probability` for the existing Android screen.

## Testing

```powershell
pytest
```

## Troubleshooting

- If `/health` shows `model_loaded: false`, confirm `trained_models/model.pkl` and `trained_models/scaler.pkl` are committed.
- If Android cannot connect to local Flask from an emulator, use `http://10.0.2.2:5000/api/v1/predict`.
- If Render starts but prediction fails, check Render logs for `model_load_failed`.
