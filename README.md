# Pulse Diagnosis

Android application with a Flask machine-learning backend for pulse diagnosis prediction.

## Structure

```text
PulseDiagnosis/
  app/                  Android app
  backend/              Flask API and ML prediction engine
    app.py              WSGI entry point
    requirements.txt    Python dependencies
    models/             Trained model artifacts
      model.pkl
      scaler.pkl
    src/                Backend source modules
    Procfile
  render.yaml           Render Blueprint, rootDir points to backend
  .env.example          Example backend environment variables
```

## Backend Model Files

This deployment uses committed model artifacts:

```text
backend/models/model.pkl
backend/models/scaler.pkl
```

Generate them locally from the backend folder:

```powershell
cd backend
python train.py
```

## Render Deployment

Use the top-level `render.yaml`. It sets:

```yaml
rootDir: backend
```

Render will install dependencies and start the API with Gunicorn.
