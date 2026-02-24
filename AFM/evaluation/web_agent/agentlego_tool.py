#!/usr/bin/env python
# coding=utf-8
import os, json, re, logging, requests
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)

SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
AGENTLEGO_TOOL_PORT = os.getenv("AGENTLEGO_TOOL_PORT", "9103")
DEFAULT_URL = f"http://{SERVER_HOST}:{AGENTLEGO_TOOL_PORT}/run"

# ------------------------------------------------------------
# Helper: parse and repair query text
# ------------------------------------------------------------
# def parse_agentlego_query(tool: str, query: str) -> Dict[str, Any]:
#     """
#     Extract image path, text, object, bbox, attribute from a natural-language query.
#     Automatically fixes hallucinated paths.
#     """
#     cleaned = str(query).strip().replace("\n", " ").replace("\r", " ")
#     payload: Dict[str, Any] = {}

#     # --- extract image path ---
#     image_pattern = r"(?:/[^ '\")]+?\.(?:png|jpg|jpeg))"
#     found_paths = re.findall(image_pattern, cleaned, flags=re.IGNORECASE)
#     img_path = found_paths[0] if found_paths else ""

#     # --- fix common path hallucinations ---
#     if "Agent Foundation Models" in img_path:
#         img_path = img_path.replace("Agent Foundation Models", "Agent_Foundation_Models")
#     if "Foundation Models" in img_path:
#         img_path = img_path.replace("Foundation Models", "Foundation_Models")
#     if "Agent_Foundation_models" in img_path:
#         img_path = img_path.replace("Agent_Foundation_models", "Agent_Foundation_Models")

#     # --- auto-complete ---
#     if os.path.basename(img_path) == img_path:
#         base_dir = "/share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/GTA/image"
#         img_path = os.path.join(base_dir, img_path)
#     elif not os.path.exists(img_path) and "/softwares/" in img_path:
#         i = img_path.find("/softwares/")
#         img_path = "/share" + img_path[i:]
#     if not os.path.exists(img_path) and " " in os.path.basename(img_path):
#         candidate = img_path.replace(" ", "_")
#         if os.path.exists(candidate):
#             img_path = candidate
#     if not os.path.exists(img_path):
#         logger.warning(f"[AgentLegoTool] Image not found; using fallback image_1.jpg")
#         img_path = "/share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/GTA/image/image_1.jpg"

#     payload["path"] = img_path

#     # --- tool-specific extractions ---
#     if tool == "text_to_bbox":
#         # e.g. "find bottle", "locate red bottle"
#         m = re.search(r"\b(?:find|detect|locate|where is)\s+([a-zA-Z0-9_ ]+)", cleaned)
#         payload["text"] = m.group(1).strip() if m else "object"

#     elif tool == "count_given_object":
#         # e.g. "count beer bottles", "how many cars"
#         m = re.search(r"\b(?:count|how many)\s+([a-zA-Z0-9_ ]+)", cleaned)
#         payload["object"] = m.group(1).strip() if m else "object"

#     elif tool == "region_attribute_description":
#         # bbox in format (x1,y1,x2,y2) or x1=.. y1=..
#         m = re.search(r"\(?\s*(\d+)[, ]+(\d+)[, ]+(\d+)[, ]+(\d+)\s*\)?", cleaned)
#         if m:
#             payload["bbox"] = [int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))]
#         else:
#             payload["bbox"] = [50, 50, 200, 200]
#         # extract attribute like "color", "shape"
#         m2 = re.search(r"\b(?:attribute|describe|color|shape)\b", cleaned, flags=re.IGNORECASE)
#         payload["attribute"] = m2.group(0).lower() if m2 else "color"

#     # OCR / ImageDescription don’t need extras
#     return payload

# DATASET_ROOTS = {
#     "GTA": "/share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/GTA/image",
#     "GAIA": "/share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/gaia/files",
#     "AGENTX": "/share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/Agent-X/files",
# }
DATASET_ROOTS = {
    "GTA": "",
    "GAIA": "/share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/gaia/files",
    "AGENTX": "/share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/Agent-X/files",
}

def parse_agentlego_query(tool: str, query: str) -> Dict[str, Any]:
    """
    Extract base filename + tool args from a natural-language query.

    Expected that the agent provides only the filename (e.g., image_1.jpg).
    No path hallucination fixing is performed here.
    """
    cleaned = str(query).strip().replace("\n", " ").replace("\r", " ")
    payload: Dict[str, Any] = {}

    # --- extract a filename (image_1.jpg etc.) ---
    # Matches things like: image_1.jpg, foo-bar.png, abc_123.jpeg
    file_pattern = r"\b([A-Za-z0-9._-]+\.(?:png|jpg|jpeg))\b"
    mfile = re.search(file_pattern, cleaned, flags=re.IGNORECASE)
    filename = mfile.group(1) if mfile else ""

    # Store just the filename for now; AgentLegoTool() will prepend dataset root
    payload["path"] = filename

    # --- tool-specific extractions ---
    if tool == "text_to_bbox":
        m = re.search(r"\b(?:find|detect|locate|where is)\s+([a-zA-Z0-9_ ]+)", cleaned)
        payload["text"] = m.group(1).strip() if m else "object"

    elif tool == "count_given_object":
        m = re.search(r"\b(?:count|how many)\s+([a-zA-Z0-9_ ]+)", cleaned)
        payload["object"] = m.group(1).strip() if m else "object"

    elif tool == "region_attribute_description":
        m = re.search(r"\(?\s*(\d+)[, ]+(\d+)[, ]+(\d+)[, ]+(\d+)\s*\)?", cleaned)
        payload["bbox"] = [int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))] if m else [50, 50, 200, 200]

        m2 = re.search(r"\b(?:attribute|describe|color|shape)\b", cleaned, flags=re.IGNORECASE)
        payload["attribute"] = m2.group(0).lower() if m2 else "color"

    return payload

# ------------------------------------------------------------
# Main tool client
# ------------------------------------------------------------
# def AgentLegoTool(
#     tool: str,
#     query: str,
#     task: str = "",
#     api_url: str = DEFAULT_URL,
#     timeout: int = 180,
# ) -> str:
#     """Client wrapper for AgentLego Tool Server with natural text parsing."""
#     try:
#         # If the query is already structured JSON, use it directly
#         try:
#             payload = json.loads(query)
#             if not isinstance(payload, dict):
#                 raise ValueError
#         except Exception:
#             payload = parse_agentlego_query(tool, query)

#         body = {"tool": tool, "query": json.dumps(payload), "task": task}
#         headers = {"Content-Type": "application/json"}

#         logger.info(f"[AgentLegoTool] Sending {tool} request to {api_url}")
#         response = requests.post(api_url, headers=headers, json=body, timeout=timeout)

#         if response.status_code != 200:
#             msg = f"[AgentLegoTool Error] HTTP {response.status_code}: {response.text[:200]}"
#             logger.warning(msg)
#             return msg

#         result = response.json()
#         if result.get("success"):
#             return result.get("output", "")
#         else:
#             err = result.get("error_message", "Unknown error")
#             logger.warning(f"[AgentLegoTool] Server returned error: {err}")
#             return f"[AgentLegoTool Error] {err}"

#     except Exception as e:
#         logger.error(f"[AgentLegoTool Exception] {e}", exc_info=True)
#         return f"[AgentLegoTool Exception] {str(e)}"

def AgentLegoTool(
    tool: str,
    query: str,
    task: str = "",
    dataset: str = "GTA",
    api_url: str = DEFAULT_URL,
    timeout: int = 180,
) -> str:
    """
    Client wrapper for AgentLego Tool Server.

    The agent provides only a base filename; we prepend the dataset root:
      - GTA  -> .../GTA/image/<filename>
      - GAIA -> .../gaia/files/<filename>
    """
    try:
        # dataset root lookup
        ds = (dataset or "GTA").upper()
        if ds not in DATASET_ROOTS:
            raise ValueError(f"Unknown dataset '{dataset}'. Use one of: {list(DATASET_ROOTS.keys())}")

        # If query is structured JSON, use it; else parse natural text
        try:
            payload = json.loads(query)
            if not isinstance(payload, dict):
                raise ValueError
        except Exception:
            payload = parse_agentlego_query(tool, query)

        # Expect payload["path"] to be just a filename; prepend dataset root
        img = payload.get("path") or payload.get("img") or payload.get("path") or ""
        img = str(img).strip()

        # If caller already passed an absolute path, keep it; otherwise prepend dataset root
        if img and not os.path.isabs(img):
            payload["path"] = os.path.join(DATASET_ROOTS[ds], img)
        else:
            payload["path"] = img

        body = {"tool": tool, "query": json.dumps(payload), "task": task}
        headers = {"Content-Type": "application/json"}

        logger.info(f"[AgentLegoTool] Sending {tool} request to {api_url} | dataset={ds}")
        response = requests.post(api_url, headers=headers, json=body, timeout=timeout)

        if response.status_code != 200:
            msg = f"[AgentLegoTool Error] HTTP {response.status_code}: {response.text[:200]}"
            logger.warning(msg)
            return msg

        result = response.json()
        if result.get("success"):
            return result.get("output", "")
        else:
            err = result.get("error_message", "Unknown error")
            logger.warning(f"[AgentLegoTool] Server returned error: {err}")
            return f"[AgentLegoTool Error] {err}"

    except Exception as e:
        logger.error(f"[AgentLegoTool Exception] {e}", exc_info=True)
        return f"[AgentLegoTool Exception] {str(e)}"

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
# if __name__ == "__main__":
    # img = "/share/softwares/haider/AgentFoundation_Models/AFM/data/web_agent/GTA/image/image_1.jpg"
    # img = "https://share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/GTA/image/image_3.jpg|"
    # print("🖼️", AgentLegoTool("image_description", f"Describe this image {img}"))
    # print("🔤", AgentLegoTool("ocr", f"OCR on {img}"))
    # print("📦", AgentLegoTool("text_to_bbox", f"find bottle in {img}"))
    # print("🍺", AgentLegoTool("count_given_object", f"count beer bottles in {img}"))
    # print("🎨", AgentLegoTool("region_attribute_description", f"Describe color in region (50,50,200,200) of {img}"))


# #!/usr/bin/env python
# # coding=utf-8
# import os
# import json
# import requests
# import logging
# from typing import Optional

# # ------------------------------------------------------------
# # Logging
# # ------------------------------------------------------------
# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
# )

# # ------------------------------------------------------------
# # Environment setup
# # ------------------------------------------------------------
# SERVER_HOST = os.getenv("SERVER_HOST", "10.127.30.114")
# AGENTLEGO_TOOL_PORT = os.getenv("AGENTLEGO_TOOL_PORT", "9400")
# DEFAULT_URL = f"http://{SERVER_HOST}:{AGENTLEGO_TOOL_PORT}/run"

# # ------------------------------------------------------------
# # Client wrapper
# # ------------------------------------------------------------
# def AgentLegoTool(
#     tool: str,
#     query: str,
#     task: str = "",
#     api_url: str = DEFAULT_URL,
#     timeout: int = 180,
# ) -> str:
#     """
#     Client wrapper for the AgentLego Tool Server (FastAPI).

#     Args:
#         tool (str): Name of the tool (e.g., "ocr", "image_description", "text_to_bbox").
#         query (str): JSON or plain string describing the image and parameters.
#         task (str): Optional context task name.
#         api_url (str): Server URL (defaults to env SERVER_HOST + AGENTLEGO_TOOL_PORT).
#         timeout (int): Timeout in seconds for requests.

#     Returns:
#         str: Output string returned by the tool server.
#     """
#     try:
#         payload = {"tool": tool, "query": query, "task": task}
#         headers = {"Content-Type": "application/json"}

#         logger.info(f"[AgentLegoTool] Sending request to {api_url} | tool={tool}")
#         response = requests.post(api_url, json=payload, headers=headers, timeout=timeout)

#         if response.status_code != 200:
#             msg = f"[AgentLegoTool Error] HTTP {response.status_code}: {response.text[:300]}"
#             logger.error(msg)
#             return msg

#         data = response.json()
#         if not data.get("success", False):
#             err = data.get("error_message", "Unknown error")
#             logger.warning(f"[AgentLegoTool Error] {err}")
#             return f"[AgentLegoTool Error] {err}"

#         return data.get("output", "")
#     except Exception as e:
#         logger.exception(f"[AgentLegoTool Exception] {e}")
#         return f"[AgentLegoTool Exception] {str(e)}"


# # ------------------------------------------------------------
# # Example usage
# # ------------------------------------------------------------
if __name__ == "__main__":
    # img = "/share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/GTA/image/image_1.jpg"
    img = "/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/image_1.jpg"
    # img = "5b2a14e8-6e59-479c-80e3-4696e8980152.jpg"

    print("\n🖼️ Image Description:")
    print(AgentLegoTool("image_description", json.dumps({"path": img})))
    # print(AgentLegoTool("image_description", json.dumps({"path": img}), dataset="GAIA"))

    print("\n🔤 OCR:")
    print(AgentLegoTool("ocr", json.dumps({"path": img})))
    # print(AgentLegoTool("ocr", json.dumps({"path": img}), dataset="GAIA"))

    print("\n📦 TextToBbox:")
    print(AgentLegoTool("text_to_bbox", json.dumps({"path": img, "text": "bottle"})))
    # print(AgentLegoTool("text_to_bbox", json.dumps({"path": img, "text": "bottle"}), dataset="GAIA"))

    print("\n🍺 CountGivenObject:")
    print(AgentLegoTool("count_given_object", json.dumps({"path": img, "object": "beer"})))
    # print(AgentLegoTool("count_given_object", json.dumps({"path": img, "object": "dog"}), dataset="GAIA"))

    print("\n🎨 RegionAttributeDescription:")
    print(AgentLegoTool("region_attribute_description", json.dumps({
        "path": img,
        "bbox": [50, 50, 200, 200],
        "attribute": "color"
    })))
    # print(AgentLegoTool("region_attribute_description", json.dumps({
    #     "path": img,
    #     "bbox": [50, 50, 200, 200],
    #     "attribute": "color"
    # }), dataset="GAIA"))

    doc_path = "3da89939-209c-4086-8520-7eb734e6b4ef.xlsx"

    print(AgentLegoTool("document_reader", json.dumps({
        "path": doc_path
    }), dataset="GAIA"))

    # zip_path = "9b54f9d9-35ee-4a14-b62f-d130ea00317f.zip"
    # print(AgentLegoTool("zip_tool", json.dumps({
    #     "path": zip_path
    # }), dataset="GAIA"))

    # py_path = "f918266a-b3e0-4914-865d-4faa564f1aef.py"
    # print(AgentLegoTool("python_interpreter", json.dumps({
    #     "path": py_path
    # }), dataset="GAIA"))

    # aud_path = "1f975693-876d-457b-a649-393859e79bf3.mp3"
    # print(AgentLegoTool("media_analysis", json.dumps({
    #     "path": aud_path
    # }), dataset="GAIA"))


    # VIDEO = "AgentX_228.mp4"
    # print(AgentLegoTool("video_metadata", json.dumps({"path": VIDEO}), dataset="AGENTX"))
    # QUESTION = "How do the individuals interact throughout the video, and what changes occur in their positions or actions over time? "
    # print(AgentLegoTool("video_description", json.dumps({"path": VIDEO, "question": QUESTION, "target_fps": 6.0, "chunk_size": 12}), dataset="AGENTX"))
    # print(AgentLegoTool("video_count_given_object", json.dumps({"path": VIDEO, "object": "police officer", "target_fps": 6.0}), dataset="AGENTX"))
    # print(AgentLegoTool("video_ocr", json.dumps({"path": VIDEO, "target_fps": 6.0}), dataset="AGENTX"))
