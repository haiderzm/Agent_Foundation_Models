import os
import sys
import json
import logging
from typing import Tuple, Dict, Any
import torch
from torch.amp import autocast
from document_reader import DocumentReader
# from zip_tool import ZipTool
from media_analysis import MediaAnalysis

# --- env tweaks ---
os.environ["MMCV_DISABLE_PROGRESSBAR"] = "1"

# --- paths ---
sys.path.insert(0, "/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/GTA/agentlego")

from agentlego.apis import load_tool
# from agentlego.tools.python_interpreter.python_interpreter import PythonInterpreter
from agentlego.tools.python_interpreter.python_interpreter import UniversalPythonInterpreter
from benchmark import ImageDescription, CountGivenObject, RegionAttributeDescription
from video_tools import (
    video_metadata as _video_metadata,
    video_description as _video_description,
    video_count_given_object as _video_count_given_object,
    video_ocr as _video_ocr,
)


# --- logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)

# canonical tool names
_CANONICAL = {
    "ocr",
    "image_description",
    "text_to_bbox",
    "count_given_object",
    "region_attribute_description",

    "document_reader",  
    # "zip_tool",
    "python_interpreter",
    "media_analysis",

    "video_metadata",
    "video_description",
    "video_count_given_object",
    "video_ocr"
}

def _mute_rich_live_once():
    """Disable rich.progress.track to avoid LiveError when multiple lives exist."""
    try:
        import rich.progress as _rp
        if getattr(_rp, "_afm_track_patched", False):
            return
        def _silent_track(seq, *args, **kwargs):
            for x in seq:
                yield x
        _rp.track = _silent_track
        _rp._afm_track_patched = True
        os.environ.setdefault("RICH_DISABLE", "1")
    except Exception:
        pass


# =====================================================================
# Manager
# =====================================================================
class AgentLegoToolManager:
    """
    Loads all AgentLego tools once (kept on GPU) and provides .run() inference.
      - OCR, TextToBbox -> agentlego.apis.load_tool()
      - ImageDescription, CountGivenObject, RegionAttributeDescription -> benchmark.py
    """
    def __init__(self, device: str = "cuda", preload: bool = True):
        self.device = device
        self.tools: Dict[str, Any] = {}
        self._ensure_nltk_assets()
        if preload:
            self._load_all()

    # -----------------------------------------------------------------
    # preload all models
    # -----------------------------------------------------------------
    def _load_all(self):
        for name in sorted(_CANONICAL):
            if name not in self.tools:
                self._load_if_needed(name)

    # -----------------------------------------------------------------
    # download NLTK assets
    # -----------------------------------------------------------------
    def _ensure_nltk_assets(self):
        try:
            import nltk
            for pkg in [
                "punkt", "punkt_tab",
                "averaged_perceptron_tagger",
                "averaged_perceptron_tagger_eng",
            ]:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"NLTK pre-download skipped: {e}")

    # -----------------------------------------------------------------
    # create a tool if not loaded yet
    # -----------------------------------------------------------------
    def _load_if_needed(self, name: str):
        if name in self.tools:
            return self.tools[name]
        if name not in _CANONICAL:
            raise ValueError(f"Unknown tool '{name}'.")

        logger.info(f"Loading '{name}' on device={self.device}")
        if name == "ocr":
            tool = load_tool("OCR", device=self.device)

        elif name == "python_interpreter":
            # tool = PythonInterpreter(timeout=60)
            tool = UniversalPythonInterpreter(timeout=120)

        elif name == "media_analysis":
            tool = MediaAnalysis()

        elif name == "text_to_bbox":
            _mute_rich_live_once()
            tool = load_tool("TextToBbox", device=self.device)
            try:
                if hasattr(tool, "_inferencer") and getattr(tool._inferencer, "visualizer", None) is not None:
                    tool._inferencer.visualizer = None
            except Exception as e:
                logger.debug(f"Could not disable visualizer: {e}")

        elif name == "image_description":
            _mute_rich_live_once()
            tool = ImageDescription(device=self.device)

        elif name == "count_given_object":
            _mute_rich_live_once()
            tool = CountGivenObject(device=self.device)

        elif name == "region_attribute_description":
            _mute_rich_live_once()
            tool = RegionAttributeDescription(device=self.device)
        
        elif name == "document_reader":
            tool = DocumentReader()

        # elif name == "zip_tool":
        #     tool = ZipTool(overwrite=True)

        elif name == "video_metadata":
            tool = _video_metadata
        elif name == "video_description":
            tool = _video_description
        elif name == "video_count_given_object":
            tool = _video_count_given_object
        elif name == "video_ocr":
            tool = _video_ocr


        self.tools[name] = tool
        logger.info(f"Loaded tool '{name}'")
        return tool
    
    def _extract_code_payload(self, query: str) -> str:
        """
        Accept either:
          - raw python markdown string
          - JSON payload: {"code": "..."} or {"command": "..."} or {"script": "..."}
        """
        q = (query or "").strip()
        if not q:
            raise ValueError("Empty query")

        try:
            payload = json.loads(q)
            if isinstance(payload, dict):
                code = payload.get("code") or payload.get("command") or payload.get("script")
                if code and isinstance(code, str):
                    return code
        except Exception:
            pass

        # fallback: treat query as direct code
        return q


    # -----------------------------------------------------------------
    # run inference
    # -----------------------------------------------------------------
    def run(self, tool_name: str, query: str, task: str = "") -> str:
        tool = self.tools.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not loaded. Call _load_all() first.")
        
        if tool_name == "python_interpreter":
            q = (query or "").strip()
            # pass through as-is; the tool handles json / markdown / raw
            return tool.apply(q)


            # allow either {"code": "..."} OR {"path": "/.../script.py"}
            # try:
            #     payload = json.loads((query or "").strip())
            # except Exception:
            #     payload = None

            # if isinstance(payload, dict) and "path" in payload:
            #     py_path = payload["path"]
            #     # reuse DocumentReader to load the script text
            #     script_lines = self.tools["document_reader"].read(py_path)  # list[str]
            #     code = "\n".join(script_lines)
            # else:
            #     code = self._extract_code_payload(query)

            # result = tool.apply(code)
            # return str(result)

        path, args = self._parse_query(query)

        # OCR in float32
        if tool_name == "ocr":
            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            try:
                result = tool(path)
            finally:
                torch.set_default_dtype(prev_dtype)
            return result
        
        if tool_name == "document_reader":
            result = tool.read(path)
            return json.dumps(result, ensure_ascii=False)
        
        # if tool_name == "zip_tool":
        #     result = tool.extract(path)
        #     return json.dumps(result, ensure_ascii=False)
        
        if tool_name == "media_analysis":
            result = tool.run(path)
            return json.dumps(result, ensure_ascii=False)
        
        # ---- video tools (API-backed; no GPU autocast needed here) ----
        if tool_name == "video_metadata":
            result = tool(path)
            return json.dumps(result, ensure_ascii=False)

        if tool_name == "video_description":
            question = args.get("question") or args.get("text") or args.get("query") or ""
            target_fps = float(args.get("target_fps", 6.0))
            chunk_size = int(args.get("chunk_size", 12))
            max_side = int(args.get("max_side", 896))
            result = tool(path, question, target_fps=target_fps, chunk_size=chunk_size, max_side=max_side)
            return json.dumps(result, ensure_ascii=False)

        if tool_name == "video_count_given_object":
            obj = args.get("object") or args.get("name") or args.get("query") or ""
            target_fps = float(args.get("target_fps", 6.0))
            max_side = int(args.get("max_side", 896))
            result = tool(path, obj, target_fps=target_fps, max_side=max_side)
            return json.dumps(result, ensure_ascii=False)

        if tool_name == "video_ocr":
            target_fps = float(args.get("target_fps", 6.0))
            max_side = int(args.get("max_side", 896))
            max_frames = int(args.get("max_frames", 24))
            result = tool(path, target_fps=target_fps, max_side=max_side, max_frames=max_frames)
            return json.dumps(result, ensure_ascii=False)


        # all others in mixed precision
        with autocast("cuda", dtype=torch.float16):
            if tool_name == "image_description":
                return tool(path)
            if tool_name == "text_to_bbox":
                text = args.get("text") or args.get("query") or args.get("caption") or ""
                top1 = bool(args.get("top1", True))
                return tool(path, text, top1)
            if tool_name == "count_given_object":
                obj = args.get("object") or args.get("name") or args.get("query") or ""
                return tool(path, obj)
            if tool_name == "region_attribute_description":
                bbox = self._to_bbox_str(args.get("bbox"))
                attr = args.get("attribute") or args.get("attr") or args.get("query") or ""
                return tool(path, bbox, attr)

        raise ValueError(f"Unsupported tool: {tool_name}")

    # -----------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------
    def _parse_query(self, query: str) -> Tuple[str, dict]:
        q = (query or "").strip()
        if not q:
            raise ValueError("Empty query")

        payload = None
        try:
            payload = json.loads(q)
            if not isinstance(payload, dict):
                payload = None
        except Exception:
            pass
        if payload is None:
            payload = {"path": q}

        img = payload.get("path") or payload.get("img") or payload.get("path")
        if not img:
            import re
            # m = re.search(r"(/[^ '\")]+?\.(?:png|jpg|jpeg))", q, flags=re.IGNORECASE)
            m = re.search(
                        # r"(/[^ '\")]+?\.(?:png|jpg|jpeg|txt|pdf|docx|pptx|csv|xlsx|json|jsonld|pdb))",
                        r"(/[^ '\")]+?\.(?:png|jpg|jpeg|txt|pdf|docx|pptx|csv|xlsx|json|jsonld|pdb|mp3|wav|m4a|mp4))",
                        q,
                        flags=re.IGNORECASE,
                    )
            if m:
                img = m.group(1)
        if not img:
            raise ValueError("No image path found in query")

        # img = self._fix_image_path(img)
        args = {k: v for k, v in payload.items() if k not in ["path", "img", "path"]}
        return img, args

    def _fix_image_path(self, p: str) -> str:
        if "Agent Foundation Models" in p:
            p = p.replace("Agent Foundation Models", "Agent_Foundation_Models")
        if "Foundation Models" in p:
            p = p.replace("Foundation Models", "Foundation_Models")
        if "Agent_Foundation_models" in p:
            p = p.replace("Agent_Foundation_models", "Agent_Foundation_Models")

        if os.path.basename(p) == p:
            base = "/share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/GTA/image"
            cand = os.path.join(base, p)
            if os.path.exists(cand):
                return cand

        if not os.path.exists(p) and (p.startswith("share/") or "/softwares/" in p):
            i = p.find("/softwares/")
            if i >= 0:
                cand = "/share" + p[i:]
                if os.path.exists(cand):
                    return cand

        if not os.path.exists(p) and " " in os.path.basename(p):
            cand = p.replace(" ", "_")
            if os.path.exists(cand):
                return cand

        return p

    def _to_bbox_str(self, bbox_any) -> str:
        if bbox_any is None:
            raise ValueError("bbox is required for region_attribute_description")
        if isinstance(bbox_any, str):
            return bbox_any.strip()
        if isinstance(bbox_any, (list, tuple)) and len(bbox_any) == 4:
            x1, y1, x2, y2 = bbox_any
            return f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
        raise ValueError(f"bbox must be 4-tuple/list or string, got: {bbox_any!r}")


# =====================================================================
# local test
# =====================================================================
if __name__ == "__main__":
    mgr = AgentLegoToolManager(device="cuda", preload=True)
    img = "/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/image_1.jpg"

    print(mgr.run("ocr", json.dumps({"path": img})))
    print(mgr.run("image_description", json.dumps({"path": img})))
    print(mgr.run("text_to_bbox", json.dumps({"path": img, "text": "bottle"})))
    print(mgr.run("count_given_object", json.dumps({"path": img, "object": "beer"})))
    print(mgr.run("region_attribute_description",
                  json.dumps({"path": img, "bbox": [50,50,200,200], "attribute": "color"})))
    
    path = "/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/3da89939-209c-4086-8520-7eb734e6b4ef.xlsx"
    print(mgr.run("document_reader", json.dumps({"path": path})))

    # zip_p = "/share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/gaia/files/9b54f9d9-35ee-4a14-b62f-d130ea00317f.zip"
    # print(mgr.run("zip_tool", json.dumps({"path": zip_p})))

    print(mgr.run("python_interpreter", json.dumps({"code": "def solution():\n    return 'ok'"})))
    print(mgr.run("python_interpreter", json.dumps({"path": "/share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/gaia/files/f918266a-b3e0-4914-865d-4faa564f1aef.py"})))

    # aud_path = "/share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/gaia/files/1f975693-876d-457b-a649-393859e79bf3.mp3"
    # print(mgr.run("media_analysis", json.dumps({"path": aud_path})))





