# Pulse Diagnosis Prediction API

Production-ready Flask backend for the Pulse Diagnosis Prediction System. The API trains a traditional machine-learning model, loads the model once at startup, predicts Normal/Abnormal pulse status, stores prediction history in SQLite, and is ready for Render deployment.

## Project Structure

```text
backend/
  app.py
  train.py
  requirements.txt
  runtime.txt
  render.yaml
  Procfile
  dataset/pulse_features.csv
  models/model.pkl
  models/scaler.pkl
  src/
    api.py
    config.py
    database.py
    evaluation.py
    feature_extractor.py
    logger.py
    predictor.py
    preprocess.py
    routes.py
    trainer.py
    utils.py
  tests/
```

## Local Setup

```powershell
cd C:\Users\Sulo\Documents\Codex\2026-07-02\act-as-a-senior-ai-engineer-2\PulseDiagnosis\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python train.py
python app.py
```

Local API: <http://127.0.0.1:5000>
Swagger UI: <http://127.0.0.1:5000/apidocs/>

## API Endpoints

### GET /health

Healthy response after model training:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "database": "connected",
  "version": "1.0.0"
}
```

If model files are missing, the service still starts and returns `model_loaded: false`. Run `python train.py` locally, or let Render run it during the build step.

### POST /predict

```json
{
  "mean": 72.5,
  "std": 4.2,
  "variance": 17.6,
  "min": 63,
  "max": 82,
  "energy": 5200
}
```

Response:

```json
{
  "prediction": "Normal",
  "confidence": 0.9821,
  "risk_level": "Low",
  "prediction_probability": {
    "Normal": 0.9821,
    "Abnormal": 0.0179
  }
}
```

Other endpoints:

- `GET /history`
- `DELETE /history`
- `GET /model-info`
- Versioned aliases: `/api/v1/predict`, `/api/v1/history`, `/api/v1/model-info`, `/api/v1/health`

## Model Artifacts for Option 2

This deployment expects trained model artifacts to be committed to GitHub. Before deploying, run:

```powershell
cd C:\Users\Sulo\Documents\Codex\2026-07-02\act-as-a-senior-ai-engineer-2\PulseDiagnosis\backend
python train.py
dir models
```

Confirm these files exist:

```text
models/model.pkl
models/scaler.pkl
models/model_metadata.json
models/metrics.json
```

Then commit them with the backend. The included `.gitignore` allows these model files to be tracked.

## Render Deployment

### 1. Push to GitHub

From the project root:

```powershell
git init
git add .
git commit -m "Prepare Pulse Diagnosis API for Render"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

### 2. Connect Render

1. Open <https://dashboard.render.com>.
2. Choose **New +**.
3. Choose **Blueprint** if using `render.yaml`, or **Web Service** for manual setup.
4. Connect the GitHub repository.
5. Select the backend directory if Render asks for the root directory: `PulseDiagnosis/backend`.

### 3. Render Settings

The included `render.yaml` defines:

- Environment: Python
- Runtime: Python 3.12.8
- Build command: installs dependencies only. Train locally and commit `models/model.pkl` plus `models/scaler.pkl` before deploying.
- Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
- Health check path: `/health`
- Auto deploy: enabled

Manual setup values:

```text
Build Command: pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
Start Command: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
Health Check Path: /health
```

## Updating Deployment

```powershell
git add .
git commit -m "Update backend"
git push
```

Render auto deploys after the push if auto deploy is enabled.

## View Logs and Restart

1. Open the Render service.
2. Go to **Logs** to view startup, prediction, and error logs.
3. Use **Manual Deploy > Clear build cache & deploy** if dependencies change.
4. Use **Restart service** for a quick restart.

## Security and Production Behavior

- Gunicorn is used in production.
- Flask `app.run()` is used only for local development.
- `PORT` is read from Render environment variables.
- Model and scaler paths are relative to the backend root by default.
- CORS is enabled and configurable with `CORS_ORIGINS`.
- Flask-Limiter protects prediction traffic.
- Request body size is limited with `MAX_CONTENT_LENGTH`.
- SQLite tables are created automatically during startup.

## Testing

```powershell
pytest
```

## Repository Root Note

If your GitHub repository root is `PulseDiagnosis`, use the root-level `render.yaml` at `PulseDiagnosis/render.yaml`. It sets `rootDir: backend` automatically. If you deploy only the backend folder as the repository root, use `backend/render.yaml`.
