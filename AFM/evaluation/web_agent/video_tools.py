from __future__ import annotations

import base64
import io
import json
import re
import shutil
import subprocess
from collections import Counter
from typing import Any, Dict, List, Optional

import cv2
import requests
from PIL import Image


# =========================
# API config
# =========================
API_URL = "http://0.0.0.0:9106/v1/generate"
API_TIMEOUT_SEC = 180
API_MAX_RETRIES = 2


def _pil_to_b64_jpeg(pil: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def api_infer(prompt: str, frames: List[Image.Image], max_new_tokens: int = 256) -> str:
    """
    Calls the local Qwen2-VL server:
      POST http://0.0.0.0:9106/v1/generate
      { "prompt": "...", "images": ["<b64>", ...], "max_new_tokens": 256 }
    Returns: response["text"]
    """
    payload = {
        "prompt": prompt,
        "images": [_pil_to_b64_jpeg(im) for im in frames],
        "max_new_tokens": max_new_tokens,
    }

    last_err: Optional[Exception] = None
    for attempt in range(API_MAX_RETRIES + 1):
        try:
            r = requests.post(API_URL, json=payload, timeout=API_TIMEOUT_SEC)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "text" in data:
                return data["text"]
            # fallback if server returns unexpected shape
            return json.dumps(data)
        except Exception as e:
            last_err = e
            if attempt < API_MAX_RETRIES:
                continue
            raise RuntimeError(f"API inference failed: {last_err}") from last_err


# =========================
# 1) Video metadata
# =========================

def _parse_frac(frac: str) -> Optional[float]:
    try:
        n, d = frac.split("/")
        n, d = float(n), float(d)
        return None if d == 0 else n / d
    except Exception:
        return None


def video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Returns: duration_sec, fps, frame_count, width, height, source
    Prefers ffprobe; falls back to OpenCV.
    """
    if shutil.which("ffprobe") is not None:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,r_frame_rate,nb_frames,duration,width,height",
            "-show_entries", "format=duration",
            "-of", "json",
            video_path,
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode == 0:
            try:
                ffj = json.loads(p.stdout)
                stream = (ffj.get("streams") or [{}])[0]
                fmt = ffj.get("format") or {}

                dur = None
                for src in (fmt.get("duration"), stream.get("duration")):
                    if src is None:
                        continue
                    try:
                        dur = float(src)
                        break
                    except Exception:
                        pass

                fps = _parse_frac(stream.get("avg_frame_rate", "")) or _parse_frac(stream.get("r_frame_rate", ""))
                fc = None
                if stream.get("nb_frames") is not None:
                    try:
                        fc = int(stream["nb_frames"])
                    except Exception:
                        pass

                return {
                    "duration_sec": dur,
                    "fps": fps,
                    "frame_count": fc,
                    "width": stream.get("width"),
                    "height": stream.get("height"),
                    "source": "ffprobe",
                }
            except Exception:
                pass

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
    dur = (float(fc) / float(fps)) if fps and fc else None
    cap.release()
    return {
        "duration_sec": dur,
        "fps": float(fps) if fps else None,
        "frame_count": fc,
        "width": w,
        "height": h,
        "source": "opencv",
    }


# =========================
# 3) Frame sampling in memory
# =========================

def _resize_bgr_max_side(frame_bgr, max_side: int):
    h, w = frame_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return frame_bgr
    scale = max_side / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def sample_frames_pil(
    video_path: str,
    target_fps: float = 6.0,
    max_side: int = 896,
) -> List[Dict[str, Any]]:
    """
    Returns a time-ordered list of sampled frames:
      [{"frame_idx": int, "t_sec": float, "pil": PIL.Image}, ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / max(target_fps, 1e-6))), 1)

    frames: List[Dict[str, Any]] = []
    i = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if i % step == 0:
            t_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or i)

            frame_bgr = _resize_bgr_max_side(frame_bgr, max_side=max_side)
            frame_rgb = frame_bgr[:, :, ::-1]
            pil = Image.fromarray(frame_rgb)

            frames.append({"frame_idx": frame_idx, "t_sec": t_sec, "pil": pil})
        i += 1

    cap.release()
    frames.sort(key=lambda x: (x["t_sec"], x["frame_idx"]))
    return frames


def _uniform_pick(n: int, k: int) -> List[int]:
    if n <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k <= 1:
        return [0]
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


# =========================
# 4) Video description (chunked multi-frame inference via API)
# =========================

def video_description(
    video_path: str,
    question: str,
    target_fps: float = 6.0,
    chunk_size: int = 12,
    max_side: int = 896,
) -> Dict[str, Any]:
    frames = sample_frames_pil(video_path, target_fps=target_fps, max_side=max_side)
    if not frames:
        return {"narrative": "No frames could be sampled.", "chunks": []}

    chunks: List[Dict[str, Any]] = []
    chunk_summaries: List[str] = []

    for start in range(0, len(frames), chunk_size):
        chunk = frames[start:start + chunk_size]
        if not chunk:
            continue

        t_start = float(chunk[0]["t_sec"])
        t_end = float(chunk[-1]["t_sec"])

        prompt = (
            "These are consecutive frames sampled from a short video segment in chronological order.\n"
            "Describe what happens over time in this segment: interactions, movements, and position changes.\n"
            "Be specific and chronological.\n\n"
            f"User question: {question}"
        )

        seg_text = api_infer(prompt, [x["pil"] for x in chunk], max_new_tokens=256)
        chunk_summaries.append(f"[{t_start:.2f}s–{t_end:.2f}s] {seg_text}")

        if len(chunk) >= 3:
            kf = [chunk[0], chunk[len(chunk) // 2], chunk[-1]]
        else:
            kf = chunk

        chunks.append({
            "chunk_id": len(chunks),
            "t_start": t_start,
            "t_end": t_end,
            "keyframes": [{"frame_idx": int(x["frame_idx"]), "t_sec": float(x["t_sec"])} for x in kf],
        })

    return {"narrative": " ".join(chunk_summaries), "chunks": chunks}


# =========================
# 5) Video count given object (Qwen-based, per-frame via API)
# =========================

def _extract_first_int(text: str) -> Optional[int]:
    nums = re.findall(r"\b\d+\b", text)
    return int(nums[0]) if nums else None


def video_count_given_object(
    video_path: str,
    object_text: str,
    target_fps: float = 6.0,
    max_side: int = 896,
) -> Dict[str, Any]:
    frames = sample_frames_pil(video_path, target_fps=target_fps, max_side=max_side)
    if not frames:
        return {"object": object_text, "max_visible_at_once": 0, "counts_over_time": []}

    prompts = [
        f"Count the exact number of {object_text} visible in this frame. Answer with only the number.",
        f"How many {object_text} are visible? Reply with only a single number.",
    ]

    counts = []
    for f in frames:
        answers: List[int] = []
        for p in prompts:
            res = api_infer(p, [f["pil"]], max_new_tokens=32).strip()
            n = _extract_first_int(res)
            if n is not None:
                answers.append(n)

        c = Counter(answers).most_common(1)[0][0] if answers else 0
        counts.append({"t_sec": float(f["t_sec"]), "frame_idx": int(f["frame_idx"]), "count": int(c)})

    max_visible = max((x["count"] for x in counts), default=0)
    return {"object": object_text, "max_visible_at_once": int(max_visible), "counts_over_time": counts}


# =========================
# 6) Video OCR (general OCR prompt via API)
# =========================

OCR_PROMPT_GENERAL = (
    "You are performing OCR on a video frame.\n\n"
    "Transcribe ONLY the text that is clearly visible and readable in the image.\n"
    "Include all visible text such as timestamps, labels, numbers, or overlays.\n\n"
    "Formatting rules:\n"
    "- Output one text element per line.\n"
    "- Do not guess or infer missing characters.\n"
    "- If no readable text is visible, output NONE."
)


def _split_lines(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln and ln.upper() != "NONE"]


def video_ocr(
    video_path: str,
    target_fps: float = 6.0,
    max_side: int = 896,
    max_frames: int = 24,
) -> Dict[str, Any]:
    frames = sample_frames_pil(video_path, target_fps=target_fps, max_side=max_side)
    if not frames:
        return {"ocr_over_time": [], "unique_lines": []}

    if len(frames) > max_frames:
        pick = _uniform_pick(len(frames), max_frames)
        frames = [frames[i] for i in pick]

    ocr_over_time = []
    seen = set()
    uniq = []

    for f in frames:
        res = api_infer(OCR_PROMPT_GENERAL, [f["pil"]], max_new_tokens=96).strip()
        lines = _split_lines(res)

        ocr_over_time.append({
            "t_sec": float(f["t_sec"]),
            "frame_idx": int(f["frame_idx"]),
            "lines": lines
        })

        for ln in lines:
            key = " ".join(ln.split())
            if key not in seen:
                seen.add(key)
                uniq.append(ln)

    return {"ocr_over_time": ocr_over_time, "unique_lines": uniq}


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    VIDEO = "/share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/Agent-X/files/AgentX_228.mp4"
    QUESTION = (
        "How do the individuals interact throughout the video, and what changes occur in their positions or actions over time? "
        "How many police officers are visible overall, and how do their positions shift throughout the video? "
        "What timestamps are visible, and what is the total duration of the scene?"
    )

    meta = video_metadata(VIDEO)
    desc = video_description(VIDEO, QUESTION, target_fps=6.0, chunk_size=12)
    cnt = video_count_given_object(VIDEO, "police officer", target_fps=6.0)
    ocrr = video_ocr(VIDEO, target_fps=6.0)

    print("META:", meta)
    print("DESC:", desc["narrative"][:400], "...")
    print("COUNT max_visible_at_once:", cnt["max_visible_at_once"])
    print("OCR unique_lines:", ocrr["unique_lines"])




# from __future__ import annotations

# import json
# import re
# import shutil
# import subprocess
# from collections import Counter
# from typing import Any, Dict, List, Optional

# import cv2
# from PIL import Image


# # =========================
# # 1) Video metadata
# # =========================

# def _parse_frac(frac: str) -> Optional[float]:
#     try:
#         n, d = frac.split("/")
#         n, d = float(n), float(d)
#         return None if d == 0 else n / d
#     except Exception:
#         return None


# def video_metadata(video_path: str) -> Dict[str, Any]:
#     """
#     Returns: duration_sec, fps, frame_count, width, height, source
#     Prefers ffprobe; falls back to OpenCV.
#     """
#     if shutil.which("ffprobe") is not None:
#         cmd = [
#             "ffprobe", "-v", "error",
#             "-select_streams", "v:0",
#             "-show_entries", "stream=avg_frame_rate,r_frame_rate,nb_frames,duration,width,height",
#             "-show_entries", "format=duration",
#             "-of", "json",
#             video_path,
#         ]
#         p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         if p.returncode == 0:
#             try:
#                 ffj = json.loads(p.stdout)
#                 stream = (ffj.get("streams") or [{}])[0]
#                 fmt = ffj.get("format") or {}

#                 dur = None
#                 for src in (fmt.get("duration"), stream.get("duration")):
#                     if src is None:
#                         continue
#                     try:
#                         dur = float(src)
#                         break
#                     except Exception:
#                         pass

#                 fps = _parse_frac(stream.get("avg_frame_rate", "")) or _parse_frac(stream.get("r_frame_rate", ""))
#                 fc = None
#                 if stream.get("nb_frames") is not None:
#                     try:
#                         fc = int(stream["nb_frames"])
#                     except Exception:
#                         pass

#                 return {
#                     "duration_sec": dur,
#                     "fps": fps,
#                     "frame_count": fc,
#                     "width": stream.get("width"),
#                     "height": stream.get("height"),
#                     "source": "ffprobe",
#                 }
#             except Exception:
#                 pass

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError(f"Could not open video: {video_path}")
#     fps = cap.get(cv2.CAP_PROP_FPS) or None
#     fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
#     dur = (float(fc) / float(fps)) if fps and fc else None
#     cap.release()
#     return {
#         "duration_sec": dur,
#         "fps": float(fps) if fps else None,
#         "frame_count": fc,
#         "width": w,
#         "height": h,
#         "source": "opencv",
#     }


# # =========================
# # 2) Qwen2-VL multi-image inferencer (PIL frames)
# # =========================

# class Qwen2VLVideoInferencer:
#     """
#     Minimal Qwen2-VL wrapper for multi-image inference.
#     Uses the same messaging + process_vision_info pipeline as your Image inferencer.
#     """

#     def __init__(self, model: str, device: str = "cuda"):
#         from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
#         from qwen_vl_utils import process_vision_info
#         import torch

#         self.device = device
#         self.model = Qwen2VLForConditionalGeneration.from_pretrained(
#             model,
#             torch_dtype=torch.float16,
#             device_map="auto",
#         )
#         self.processor = AutoProcessor.from_pretrained(model)
#         self.process_vision_info = process_vision_info

#     @staticmethod
#     def _resize_pil(pil_image: Image.Image, max_side: int) -> Image.Image:
#         w, h = pil_image.size
#         if max(w, h) <= max_side:
#             return pil_image
#         scale = max_side / float(max(w, h))
#         new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
#         return pil_image.resize((new_w, new_h), resample=Image.BICUBIC)

#     def infer(
#         self,
#         frames: List[Image.Image],
#         prompt: str,
#         max_side: int = 896,
#         max_new_tokens: int = 256,
#     ) -> str:
#         import torch

#         pil_images = [self._resize_pil(im, max_side) for im in frames]

#         messages = [{
#             "role": "user",
#             "content": [{"type": "text", "text": prompt}] +
#                        [{"type": "image", "image": im} for im in pil_images],
#         }]

#         text_prompt = self.processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )

#         image_inputs, video_inputs = self.process_vision_info(messages)

#         inputs = self.processor(
#             text=[text_prompt],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         ).to(self.device)

#         with torch.no_grad():
#             generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

#         trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated)]
#         out = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#         return out[0]


# # =========================
# # 3) Frame sampling in memory
# # =========================

# def _resize_bgr_max_side(frame_bgr, max_side: int):
#     h, w = frame_bgr.shape[:2]
#     m = max(h, w)
#     if m <= max_side:
#         return frame_bgr
#     scale = max_side / float(m)
#     new_w = max(1, int(round(w * scale)))
#     new_h = max(1, int(round(h * scale)))
#     return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


# def sample_frames_pil(
#     video_path: str,
#     target_fps: float = 6.0,
#     max_side: int = 896,
# ) -> List[Dict[str, Any]]:
#     """
#     Returns a time-ordered list of sampled frames:
#       [{"frame_idx": int, "t_sec": float, "pil": PIL.Image}, ...]
#     """
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError(f"Could not open video: {video_path}")

#     src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     step = max(int(round(src_fps / max(target_fps, 1e-6))), 1)

#     frames: List[Dict[str, Any]] = []
#     i = 0
#     while True:
#         ok, frame_bgr = cap.read()
#         if not ok:
#             break
#         if i % step == 0:
#             t_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
#             frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or i)

#             frame_bgr = _resize_bgr_max_side(frame_bgr, max_side=max_side)
#             frame_rgb = frame_bgr[:, :, ::-1]
#             pil = Image.fromarray(frame_rgb)

#             frames.append({"frame_idx": frame_idx, "t_sec": t_sec, "pil": pil})
#         i += 1

#     cap.release()
#     frames.sort(key=lambda x: (x["t_sec"], x["frame_idx"]))
#     return frames


# def _uniform_pick(n: int, k: int) -> List[int]:
#     if n <= 0:
#         return []
#     if k >= n:
#         return list(range(n))
#     if k <= 1:
#         return [0]
#     return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


# # =========================
# # 4) Video description (chunked multi-frame inference)
# # =========================

# def video_description(
#     video_path: str,
#     question: str,
#     inferencer: Qwen2VLVideoInferencer,
#     target_fps: float = 6.0,
#     chunk_size: int = 12,
#     max_side: int = 896,
# ) -> Dict[str, Any]:
#     """
#     Returns:
#       {
#         "narrative": str,
#         "chunks": [
#           {"chunk_id": int, "t_start": float, "t_end": float,
#            "keyframes": [{"frame_idx": int, "t_sec": float}, ...]
#           }, ...
#         ]
#       }
#     """
#     frames = sample_frames_pil(video_path, target_fps=target_fps, max_side=max_side)
#     if not frames:
#         return {"narrative": "No frames could be sampled.", "chunks": []}

#     chunks: List[Dict[str, Any]] = []
#     chunk_summaries: List[str] = []

#     for start in range(0, len(frames), chunk_size):
#         chunk = frames[start:start + chunk_size]
#         if not chunk:
#             continue

#         t_start = float(chunk[0]["t_sec"])
#         t_end = float(chunk[-1]["t_sec"])

#         prompt = (
#             "These are consecutive frames sampled from a short video segment in chronological order.\n"
#             "Describe what happens over time in this segment: interactions, movements, and position changes.\n"
#             "Be specific and chronological.\n\n"
#             f"User question: {question}"
#         )

#         seg_text = inferencer.infer([x["pil"] for x in chunk], prompt, max_side=max_side, max_new_tokens=256)
#         chunk_summaries.append(f"[{t_start:.2f}s–{t_end:.2f}s] {seg_text}")

#         # Evidence: first/middle/last frame of chunk
#         if len(chunk) >= 3:
#             kf = [chunk[0], chunk[len(chunk) // 2], chunk[-1]]
#         else:
#             kf = chunk

#         chunks.append({
#             "chunk_id": len(chunks),
#             "t_start": t_start,
#             "t_end": t_end,
#             "keyframes": [{"frame_idx": int(x["frame_idx"]), "t_sec": float(x["t_sec"])} for x in kf],
#         })

#     return {"narrative": " ".join(chunk_summaries), "chunks": chunks}


# # =========================
# # 5) Video count given object (Qwen-based, per-frame)
# # =========================

# def _extract_first_int(text: str) -> Optional[int]:
#     nums = re.findall(r"\b\d+\b", text)
#     return int(nums[0]) if nums else None


# def video_count_given_object(
#     video_path: str,
#     object_text: str,
#     inferencer: Qwen2VLVideoInferencer,
#     target_fps: float = 6.0,
#     max_side: int = 896,
# ) -> Dict[str, Any]:
#     """
#     Returns:
#       {
#         "object": str,
#         "max_visible_at_once": int,
#         "counts_over_time": [{"t_sec": float, "frame_idx": int, "count": int}, ...]
#       }
#     """
#     frames = sample_frames_pil(video_path, target_fps=target_fps, max_side=max_side)
#     if not frames:
#         return {"object": object_text, "max_visible_at_once": 0, "counts_over_time": []}

#     prompts = [
#         f"Count the exact number of {object_text} visible in this frame. Answer with only the number.",
#         f"How many {object_text} are visible? Reply with only a single number.",
#     ]

#     counts = []
#     for f in frames:
#         answers: List[int] = []
#         for p in prompts:
#             res = inferencer.infer([f["pil"]], p, max_side=max_side, max_new_tokens=32).strip()
#             n = _extract_first_int(res)
#             if n is not None:
#                 answers.append(n)
#         c = Counter(answers).most_common(1)[0][0] if answers else 0
#         counts.append({"t_sec": float(f["t_sec"]), "frame_idx": int(f["frame_idx"]), "count": int(c)})

#     max_visible = max((x["count"] for x in counts), default=0)
#     return {"object": object_text, "max_visible_at_once": int(max_visible), "counts_over_time": counts}


# # =========================
# # 6) Video OCR (general OCR prompt, Qwen-based)
# # =========================

# OCR_PROMPT_GENERAL = (
#     "You are performing OCR on a video frame.\n\n"
#     "Transcribe ONLY the text that is clearly visible and readable in the image.\n"
#     "Include all visible text such as timestamps, labels, numbers, or overlays.\n\n"
#     "Formatting rules:\n"
#     "- Output one text element per line.\n"
#     "- Do not guess or infer missing characters.\n"
#     "- If no readable text is visible, output NONE."
# )


# def _split_lines(text: str) -> List[str]:
#     lines = [ln.strip() for ln in text.splitlines()]
#     return [ln for ln in lines if ln and ln.upper() != "NONE"]


# def video_ocr(
#     video_path: str,
#     inferencer: Qwen2VLVideoInferencer,
#     target_fps: float = 6.0,
#     max_side: int = 896,
#     max_frames: int = 24,
# ) -> Dict[str, Any]:
#     """
#     Returns:
#       {
#         "ocr_over_time": [{"t_sec": float, "frame_idx": int, "lines": [str, ...]}, ...],
#         "unique_lines": [str, ...]
#       }
#     """
#     frames = sample_frames_pil(video_path, target_fps=target_fps, max_side=max_side)
#     if not frames:
#         return {"ocr_over_time": [], "unique_lines": []}

#     # Limit number of frames we OCR (2–6s clips: this is plenty)
#     if len(frames) > max_frames:
#         pick = _uniform_pick(len(frames), max_frames)
#         frames = [frames[i] for i in pick]

#     ocr_over_time = []
#     seen = set()
#     uniq = []

#     for f in frames:
#         res = inferencer.infer([f["pil"]], OCR_PROMPT_GENERAL, max_side=max_side, max_new_tokens=96).strip()
#         lines = _split_lines(res)

#         ocr_over_time.append({
#             "t_sec": float(f["t_sec"]),
#             "frame_idx": int(f["frame_idx"]),
#             "lines": lines
#         })

#         for ln in lines:
#             key = " ".join(ln.split())
#             if key not in seen:
#                 seen.add(key)
#                 uniq.append(ln)

#     return {"ocr_over_time": ocr_over_time, "unique_lines": uniq}


# # =========================
# # Example usage
# # =========================
# if __name__ == "__main__":
#     MODEL = "/share/users/md_zama/hf_cache/Qwen2VL7B"
#     VIDEO = "/share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/Agent-X/files/AgentX_228.mp4"
#     QUESTION = (
#         "How do the individuals interact throughout the video, and what changes occur in their positions or actions over time? "
#         "How many police officers are visible overall, and how do their positions shift throughout the video? "
#         "What timestamps are visible, and what is the total duration of the scene?"
#     )

#     inf = Qwen2VLVideoInferencer(model=MODEL, device="cuda")

#     meta = video_metadata(VIDEO)
#     desc = video_description(VIDEO, QUESTION, inf, target_fps=6.0, chunk_size=12)
#     cnt = video_count_given_object(VIDEO, "police officer", inf, target_fps=6.0)
#     ocrr = video_ocr(VIDEO, inf, target_fps=6.0)

#     print("META:", meta)
#     print("DESC:", desc["narrative"][:400], "...")
#     print("COUNT max_visible_at_once:", cnt["max_visible_at_once"])
#     print("OCR unique_lines:", ocrr["unique_lines"])
