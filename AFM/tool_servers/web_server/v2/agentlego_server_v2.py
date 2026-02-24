#!/usr/bin/env python
# coding=utf-8
"""
AgentLego Tool Server V2
Serves preloaded visual tools (OCR, ImageDescription, TextToBbox, etc.)
via FastAPI for multi-agent use.
"""

import os
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import json

import sys, os
sys.path.insert(0, "/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/AFM/evaluation/web_agent")

from agentlego_manager import AgentLegoToolManager

# ---------------------------------------------------------------------
# logging setup
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# env config
# ---------------------------------------------------------------------
PORT = int(os.getenv("AGENTLEGO_TOOL_PORT", "9400"))
DEVICE = os.getenv("AGENTLEGO_DEVICE", "cuda")

# ---------------------------------------------------------------------
# initialize the manager once (models load now)
# ---------------------------------------------------------------------
logger.info(f"Initializing AgentLegoToolManager on {DEVICE} (preload=True)")
manager = AgentLegoToolManager(device=DEVICE, preload=True)

# ---------------------------------------------------------------------
# pydantic models
# ---------------------------------------------------------------------
class ToolRequest(BaseModel):
    tool: str = Field(..., description="Tool name: ocr, image_description, text_to_bbox, etc.")
    query: str = Field(..., description="JSON or plain string query for the tool")
    task: Optional[str] = Field("", description="Optional task context")

class ToolResponse(BaseModel):
    success: bool
    tool: str
    output: str
    error_message: Optional[str] = None
    processing_time: float

# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AgentLegoToolServer starting up ...")
    yield
    logger.info("AgentLegoToolServer shutting down ...")

app = FastAPI(
    title="AgentLego Tool Server V2",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------
@app.post("/run", response_model=ToolResponse)
async def run_tool(req: ToolRequest):
    start = time.time()
    try:
        loop = asyncio.get_event_loop()
        # run the synchronous .run() in thread executor
        # output = await loop.run_in_executor(None, manager.run, req.tool, req.query, req.task)
        # elapsed = time.time() - start
        # return ToolResponse(success=True, tool=req.tool, output=output, processing_time=elapsed)
        output = await loop.run_in_executor(None, manager.run, req.tool, req.query, req.task)

        # Always stringify output to be JSON-safe
        try:
            if not isinstance(output, str):
                output = json.dumps(output) if not isinstance(output, (int, float, str)) else str(output)
        except Exception:
            output = str(output)

        elapsed = time.time() - start
        return ToolResponse(success=True, tool=req.tool, output=output, processing_time=elapsed)

    except Exception as e:
        elapsed = time.time() - start
        logger.exception(f"[AgentLegoToolServer] Error in /run: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "loaded_tools": list(manager.tools.keys()),
        "time": time.time(),
    }

@app.get("/")
async def root():
    return {
        "message": "AgentLego Tool Server V2",
        "version": "1.0.0",
        "endpoints": ["/run", "/health", "/docs"],
    }

# ---------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logger.info(f"Starting AgentLego Tool Server on port {PORT}")
    uvicorn.run("agentlego_server_v2:app", host="0.0.0.0", port=PORT, workers=1)
