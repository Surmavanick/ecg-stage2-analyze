from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import tempfile

app = FastAPI(
    title="ECG Analysis API",
    description="Stage 2 - Image to Signal & Simple Features (no scipy)",
    version="0.3.1"
)

@app.get("/")
async def root():
    return {"ok": True, "message": "ECG API alive"}

@app.post("/analyze")
async def analyze_ecg(file: UploadFile = File(...)):
    """
    Step 1: იღებს ECG trace.png
    Step 2: გარდაქმნის signal vector-ად
    Step 3: ამოიცნობს პიკებს numpy-ით და ითვლის HR
    """
    # დროებითი ფაილის შენახვა
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # OpenCV - grayscale
    img = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)

    # invert
    img = cv2.bitwise_not(img)

    # Image → Signal
    signal = []
    for x in range(img.shape[1]):
        column = img[:, x]
        y_positions = np.where(column > 200)[0]
        if len(y_positions) > 0:
            y_mean = int(np.mean(y_positions))
            signal.append(y_mean)
        else:
            signal.append(None)

    # clean signal
    signal = np.array([s for s in signal if s is not None])
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    # --- Simple peak detection with numpy ---
    peaks = []
    min_distance = 30  # მინიმალური დაშორება პიკებს შორის (samples)
    threshold = 0.5    # მხოლოდ მაღალი მნიშვნელობები

    last_peak = -min_distance
    for i in range(1, len(signal)-1):
        if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            if (i - last_peak) >= min_distance:
                peaks.append(i)
                last_peak = i

    peaks = np.array(peaks)

    # RR intervals & HR
    rr_intervals = np.diff(peaks)
    heart_rate = None
    rr_variability = None

    if len(rr_intervals) > 0:
        fs = 250  # ვივარაუდოთ 250 Hz sampling
        rr_sec = rr_intervals / fs
        mean_rr = np.mean(rr_sec)
        heart_rate = 60.0 / mean_rr
        rr_variability = float(np.std(rr_sec) / mean_rr)

    return JSONResponse(content={
        "ok": True,
        "filename": file.filename,
        "signal_length": len(signal),
        "heart_rate": heart_rate,
        "rr_variability": rr_variability,
        "num_peaks": len(peaks),
        "rr_intervals_samples": rr_intervals[:10].tolist() if len(rr_intervals) > 0 else []
    })
