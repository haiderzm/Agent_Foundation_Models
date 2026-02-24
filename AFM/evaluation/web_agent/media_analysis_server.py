from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os

from faster_whisper import WhisperModel

app = FastAPI(title="MediaAnalysis Service")

# -------------------------
# Load Whisper ONCE
# -------------------------
model = WhisperModel(
    "base",
    device="cpu",          # or "cpu"
    compute_type="int8"  # or "int8" on cpu
)

# -------------------------
# Request / Response schema
# -------------------------
class MediaRequest(BaseModel):
    file_path: str


class SegmentItem(BaseModel):
    start: float
    end: float
    text: str


class MediaResponse(BaseModel):
    type: str
    path: str
    language: Optional[str]
    duration_s: float
    text: str
    segments: List[SegmentItem]


# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=MediaResponse)
def analyze_media(req: MediaRequest):
    if not os.path.exists(req.file_path):
        return MediaResponse(
            type="audio",
            path=req.file_path,
            language=None,
            duration_s=0.0,
            text="",
            segments=[],
        )

    segments_iter, info = model.transcribe(
        req.file_path,
        beam_size=5,
        vad_filter=True,
    )

    segments: List[SegmentItem] = []
    full_text: List[str] = []

    for seg in segments_iter:
        text = (seg.text or "").strip()
        segments.append(
            SegmentItem(
                start=float(seg.start),
                end=float(seg.end),
                text=text,
            )
        )
        if text:
            full_text.append(text)

    return MediaResponse(
        type="audio",
        path=req.file_path,
        language=info.language,
        duration_s=float(info.duration or 0.0),
        text=" ".join(full_text),
        segments=segments,
    )
