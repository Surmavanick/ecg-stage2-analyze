from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import tempfile

app = FastAPI(
    title="ECG Analysis API",
    description="Stage 2 - Image to Signal & Features with improved peak detection",
    version="0.4.0"
)

@app.get("/")
async def root():
    return {"ok": True, "message": "ECG API alive"}

@app.post("/analyze")
async def analyze_ecg(file: UploadFile = File(...)):
    """
    Step 1: იღებს ECG trace.png
    Step 2: გარდაქმნის signal vector-ად
    Step 3: ამოიცნობს პიკებს (improved detection)
    Step 4: ითვლის HR და Variability
    """
    # დროებითი ფაილის შენახვა
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # OpenCV - grayscale
    img = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.bitwise_not(img)  # invert

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

    # --- Step A: smoothing ---
    window = 5
    signal_smooth = np.convolve(signal, np.ones(window)/window, mode="same")

    # --- Step B: stricter peak detection ---
    peaks = []
    min_distance = 40   # რამდენ sample უნდა იყოს მინიმუმ შორის
    threshold = 0.6     # მინიმუმ სიმაღლე
    last_peak = -min_distance

    for i in range(1, len(signal_smooth)-1):
        if (
            signal_smooth[i] > threshold
            and signal_smooth[i] > signal_smooth[i-1]
            and signal_smooth[i] > signal_smooth[i+1]
        ):
            if (i - last_peak) >= min_distance:
                peaks.append(i)
                last_peak = i

    # --- Step C: remove false peaks (too close) ---
    filtered_peaks = []
    for i in range(len(peaks)):
        if i == 0 or (peaks[i] - peaks[i-1]) > 100:
            filtered_peaks.append(peaks[i])
    peaks = np.array(filtered_peaks)

    # RR intervals & HR
    rr_intervals = np.diff(peaks)
    heart_rate = None
    rr_variability = None

    if len(rr_intervals) > 0:
        fs = 250  # placeholder sampling rate
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
