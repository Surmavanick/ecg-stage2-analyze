# ECG Analysis API (Stage 2 Skeleton)

## Overview
Minimal FastAPI project for ECG arrhythmia analysis pipeline.  
Stage 2 starts here: Image → Signal → Features → Diagnosis.

## Run locally
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints
- `GET /` → healthcheck
- `POST /analyze` → accepts `trace.png` file upload
