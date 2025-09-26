from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI(
    title="ECG Analysis API",
    description="Stage 2 - Image to Signal & Arrhythmia detection (skeleton)",
    version="0.1.0"
)

@app.get("/")
async def root():
    return {"ok": True, "message": "ECG API alive"}

@app.post("/analyze")
async def analyze_ecg(file: UploadFile = File(...)):
    """
    Placeholder endpoint: accepts ECG trace.png
    """
    return JSONResponse(content={
        "ok": True,
        "filename": file.filename,
        "status": "received",
        "next_step": "convert image to signal"
    })
