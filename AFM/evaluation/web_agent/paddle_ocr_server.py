from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import cv2
from paddleocr import PaddleOCR
import os

app = FastAPI(title="PaddleOCR Service")

# -------------------------
# Load OCR ONCE
# -------------------------
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    # use_gpu=True
)

# -------------------------
# Request / Response schema
# -------------------------
class OCRRequest(BaseModel):
    image_path: str


class OCRItem(BaseModel):
    bbox: List[int]   # [x1, y1, x2, y2]
    text: str
    score: float


class OCRResponse(BaseModel):
    results: List[OCRItem]


# -------------------------
# Utils
# -------------------------
def poly_to_bbox(poly: np.ndarray):
    xs = poly[:, 0]
    ys = poly[:, 1]
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max())
    ]


# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr", response_model=OCRResponse)
def run_ocr(req: OCRRequest):
    if not os.path.exists(req.image_path):
        return {"results": []}

    result = ocr.predict(input=req.image_path)[0]

    texts = result["rec_texts"]
    polys = result["rec_polys"]
    scores = result["rec_scores"]

    items = []
    for text, poly, score in zip(texts, polys, scores):
        bbox = poly_to_bbox(np.array(poly))
        items.append(
            OCRItem(
                bbox=bbox,
                text=text,
                score=float(score)
            )
        )

    # sort top-to-bottom (y1)
    items.sort(key=lambda x: x.bbox[1])

    return OCRResponse(results=items)
