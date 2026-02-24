import os
from typing import Dict, Any, List, Optional

# from faster_whisper import WhisperModel
import requests

# class MediaAnalysis:
#     """
#     Minimal media analyzer.
#     Currently supports:
#       - Audio / video -> speech-to-text

#     Supported extensions:
#       .mp3, .wav, .m4a, .mp4
#     """

#     AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".mp4"}

#     def __init__(
#         self,
#         model_size: str = "base",
#         device: str = "cpu",
#         compute_type: str = "int8",
#     ):
#         self.model = WhisperModel(
#             model_size,
#             device=device,
#             compute_type=compute_type,
#         )

#     def run(self, file_path: str) -> Dict[str, Any]:
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(file_path)

#         ext = os.path.splitext(file_path)[1].lower()

#         if ext in self.AUDIO_EXTS:
#             return self._audio_to_text(file_path)

#         raise ValueError(f"Unsupported media type: {ext}")

#     def _audio_to_text(self, file_path: str) -> Dict[str, Any]:
#         """
#         Returns only data present in the media:
#           - full transcript
#           - timestamped segments
#         """
#         segments_iter, info = self.model.transcribe(
#             file_path,
#             beam_size=5,
#             vad_filter=True,
#         )

#         segments: List[Dict[str, Any]] = []
#         full_text: List[str] = []

#         for seg in segments_iter:
#             text = (seg.text or "").strip()
#             segments.append(
#                 {
#                     "start": float(seg.start),
#                     "end": float(seg.end),
#                     "text": text,
#                 }
#             )
#             if text:
#                 full_text.append(text)

#         return {
#             "type": "audio",
#             "path": file_path,
#             "language": info.language,
#             "duration_s": float(info.duration or 0.0),
#             "text": " ".join(full_text),
#             "segments": segments,
#         }

class MediaAnalysis:
    AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".mp4"}

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1/analyze",
        timeout: int = 300,
    ):
        self.endpoint = endpoint
        self.timeout = timeout

    def run(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.AUDIO_EXTS:
            raise ValueError(f"Unsupported media type: {ext}")

        resp = requests.post(
            self.endpoint,
            json={"file_path": file_path},
            timeout=self.timeout,
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"MediaAnalysis server error {resp.status_code}: {resp.text}"
            )

        return resp.json()

# analyzer = MediaAnalysis()

# fp = "/share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/gaia/files/1f975693-876d-457b-a649-393859e79bf3.mp3"
# result = analyzer.run(fp)

# print(result["text"])