from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import tempfile

app = FastAPI(
    title="ECG Analysis API",
    description="Stage 2 - Simple Image to Signal extraction",
    version="0.2.0"
)

@app.get("/")
async def root():
    return {"ok": True, "message": "ECG API alive"}

@app.post("/analyze")
async def analyze_ecg(file: UploadFile = File(...)):
    """
    Step 1: იღებს ECG trace.png
    Step 2: გარდაქმნის მარტივ signal vector-ად
    """
    # დროებითი ფაილის შენახვა
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # OpenCV - ვკითხულობთ grayscale-ად
    img = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)

    # Invert (ხაზი გავხადოთ თეთრი შავ ფონზე)
    img = cv2.bitwise_not(img)

    # თითო სვეტში ვპოულობთ brightest pixel-ის y-ს (ეს ხაზის პოზიციაა)
    signal = []
    for x in range(img.shape[1]):
        column = img[:, x]
        y_positions = np.where(column > 200)[0]  # bright pixels
        if len(y_positions) > 0:
            y_mean = int(np.mean(y_positions))
            signal.append(y_mean)
        else:
            signal.append(None)

    # Normalize (optional: წაშალე თუ არ გინდა)
    signal = np.array([s for s in signal if s is not None])
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    return JSONResponse(content={
        "ok": True,
        "filename": file.filename,
        "signal_length": len(signal),
        "signal_preview": signal[:20].tolist()  # პირველი 20 sample
    })
