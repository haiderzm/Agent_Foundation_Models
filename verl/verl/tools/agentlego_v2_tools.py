import asyncio
import json
import logging
import os
from typing import Any, Dict, Tuple, Optional

import aiohttp

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DATASET_ROOTS = {
    "GTA": "/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models",
    "GAIA": "/share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/gaia/files",
    "AGENTX": "/share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/Agent-X/files",
}

# If you want fallback parsing identical to your existing script, you can import it:
# from agentlego_tool import parse_agentlego_query
# For now, keep it strict: expect JSON from SFT (recommended).
def _safe_json_loads(s: str) -> Optional[dict]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


class _AgentLegoSingleTool(BaseTool):
    """
    Base class: one OpenAI tool name maps to one server tool name.
    Subclasses MUST set TOOL_NAME.
    """
    TOOL_NAME: str = ""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema = None):
        super().__init__(config, tool_schema)
        # self._instance_dict: Dict[str, Dict[str, Any]] = {}
        self._instance_dict = {}
        super().__init__(config, tool_schema)

        server_host = os.environ.get("SERVER_HOST")
        if not server_host:
            raise ValueError("SERVER_HOST is not set")

        port = int(os.environ.get("AGENTLEGO_TOOL_PORT", str(config.get("port", 9400))))
        self.endpoint = f"http://{server_host}:{port}/run"

        self.dataset = (config.get("dataset") or os.environ.get("DATASET_EVAL") or "GTA").upper()
        if self.dataset not in DATASET_ROOTS:
            raise ValueError(f"Unknown dataset '{self.dataset}'. Expected one of: {list(DATASET_ROOTS.keys())}")

        self.timeout_s = float(config.get("timeout_s", 180))
        self.max_retries = int(config.get("max_retries", 1))

        if not self.TOOL_NAME:
            raise ValueError("TOOL_NAME is empty in subclass")

        logger.info(f"Initialized AgentLego tool={self.TOOL_NAME} endpoint={self.endpoint} dataset={self.dataset}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        existing = getattr(self, "tool_schema", None)
        if existing is not None:
            return existing

        # IMPORTANT: function name must match the tag name the SFT model outputs.
        return OpenAIFunctionToolSchema(
            type="function",
            function={
                "name": self.TOOL_NAME,
                "description": f"AgentLego tool: {self.TOOL_NAME}. Input is a JSON string payload.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "JSON string tool input. Example: "
                                "{\"path\":\"image_1.jpg\"} plus tool-specific fields. "
                                "Path may be a base filename; it will be resolved using dataset root."
                            ),
                        }
                    },
                    "required": ["query"],
                },
            },
        )

    async def create(self, instance_id: str, **kwargs) -> bool:
        self._instance_dict[instance_id] = {"calls": 0}
        return True

    def _infer_task_from_messages(self, kwargs: Dict[str, Any]) -> str:
        msgs = kwargs.get("_messages_list_of_dic")
        if not msgs:
            return ""
        if len(msgs) > 1 and msgs[1].get("role") == "user":
            return msgs[1].get("content", "") or ""
        return ""

    def _resolve_path(self, payload: dict) -> dict:
        """
        If payload['path'] is a filename (not absolute), prepend dataset root.
        Mirrors your existing AgentLegoTool() behavior.
        """
        p = payload.get("path") or payload.get("img") or ""
        p = str(p).strip()
        if p and not os.path.isabs(p):
            payload["path"] = os.path.join(DATASET_ROOTS[self.dataset], p)
        else:
            payload["path"] = p
        return payload

    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        query = parameters.get("query")
        if not query or not isinstance(query, str):
            err = "Error: 'query' is missing, empty, or not a string."
            logger.error(f"[{self.TOOL_NAME}] {err} parameters={parameters}")
            return json.dumps({"error": err}), 0.0, {"error": "invalid_parameters"}

        task = self._infer_task_from_messages(kwargs)

        payload = _safe_json_loads(query)
        if payload is None:
            # If you truly never expect non-JSON (since SFT emits JSON), fail fast:
            err = f"Error: query for {self.TOOL_NAME} must be a JSON object string."
            logger.error(f"[{self.TOOL_NAME}] {err} query={query[:200]}")
            return json.dumps({"error": err}), 0.0, {"error": "invalid_query_format"}

        payload = self._resolve_path(payload)

        body = {
            "tool": self.TOOL_NAME,
            "query": json.dumps(payload, ensure_ascii=False),
            "task": task or "",
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        headers = {"Content-Type": "application/json"}

        st = self._instance_dict.setdefault(instance_id, {"calls": 0})
        st["calls"] += 1

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.endpoint, json=body, headers=headers, timeout=timeout) as resp:
                        text = await resp.text()
                        if resp.status != 200:
                            last_error = f"HTTP {resp.status}: {text[:800]}"
                            raise RuntimeError(last_error)
                        result = await resp.json()

                if not result.get("success", True):
                    last_error = result.get("error_message", "server_processing_error")
                    raise RuntimeError(last_error)

                output = result.get("output", "")
                metrics = {
                    "tool": self.TOOL_NAME,
                    "dataset": self.dataset,
                    "attempt": attempt,
                    "calls": st["calls"],
                    "processing_time": result.get("processing_time", None),
                    "output_len": len(output) if isinstance(output, str) else None,
                }

                st["last_response"] = output
                st["last_metrics"] = metrics
                return output, 0.0, metrics

            except (asyncio.TimeoutError, aiohttp.ClientError, Exception) as e:
                last_error = str(e)
                logger.warning(f"[{self.TOOL_NAME}] attempt {attempt}/{self.max_retries} failed: {last_error}")

        return json.dumps({"error": last_error}), 0.0, {"error": "tool_failed", "message": last_error, "tool": self.TOOL_NAME}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        st = self._instance_dict.get(instance_id, {})
        resp = st.get("last_response", "")
        if isinstance(resp, str) and resp:
            if resp.lstrip().startswith("{") and '"error"' in resp[:200]:
                return 0.0
            return 1.0 if len(resp) > 20 else 0.5
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)


# ---- Subclasses with names matching SFT tags ----
class OCRTool(_AgentLegoSingleTool):
    TOOL_NAME = "ocr"

class ImageDescriptionTool(_AgentLegoSingleTool):
    TOOL_NAME = "image_description"

class TextToBboxTool(_AgentLegoSingleTool):
    TOOL_NAME = "text_to_bbox"

class CountGivenObjectTool(_AgentLegoSingleTool):
    TOOL_NAME = "count_given_object"

class RegionAttributeDescriptionTool(_AgentLegoSingleTool):
    TOOL_NAME = "region_attribute_description"

# You can add the rest similarly:
class DocumentReaderTool(_AgentLegoSingleTool):
    TOOL_NAME = "document_reader"

class PythonInterpreterTool(_AgentLegoSingleTool):
    TOOL_NAME = "python_interpreter"

class MediaAnalysisTool(_AgentLegoSingleTool):
    TOOL_NAME = "media_analysis"

class VideoMetadataTool(_AgentLegoSingleTool):
    TOOL_NAME = "video_metadata"

class VideoDescriptionTool(_AgentLegoSingleTool):
    TOOL_NAME = "video_description"

class VideoCountGivenObjectTool(_AgentLegoSingleTool):
    TOOL_NAME = "video_count_given_object"

class VideoOCRTool(_AgentLegoSingleTool):
    TOOL_NAME = "video_ocr"