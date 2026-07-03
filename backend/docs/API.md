# Pulse Diagnosis API

## Endpoints

- GET / - service metadata
- GET /health - health, model, and database status
- GET /version - API and model version
- POST /predict - prediction endpoint
- GET /metrics - training metrics JSON
- GET /docs - redirects to Swagger UI
- GET /swagger - redirects to Swagger UI
- GET /history - prediction history
- DELETE /history - clear prediction history

Android-compatible aliases remain available under /api/v1.

## Prediction Request

```json
{"mean":72.5,"std":4.2,"variance":17.6,"min":63,"max":82,"energy":5200}
```

## Prediction Response

```json
{
  "prediction": "Normal",
  "confidence": 0.98,
  "probability": {"Normal": 0.98, "Abnormal": 0.02},
  "prediction_probability": {"Normal": 0.98, "Abnormal": 0.02},
  "risk_level": "Low",
  "model_version": "1.0.0",
  "processing_time_ms": 3.2,
  "timestamp": "2026-07-03T00:00:00+00:00",
  "http_status": 200,
  "explanation": "..."
}
```
