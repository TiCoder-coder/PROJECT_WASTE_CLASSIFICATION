from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from detector import run_detection

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "backend ok 🚀", "message": "YOLO + SAM2 backend is running!"}


@app.post("/detect")
async def detect_api(file: UploadFile = File(...)):
    image_bytes = await file.read()

    npimg = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    detections = run_inference(frame)

    return {"detections": detections}

