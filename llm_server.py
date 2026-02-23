"""
llm_server.py — FastAPI server wrapping the xLAM-2-3b function-calling model.

Endpoints
---------
POST /chat
    Body : {"prompt": "<user message>", "stream": false}
    Returns: {
        "thinking":   "<any preamble text before the main response>",
        "expression": "<plain-text reply, empty if tool calls were made>",
        "tool_calls": [{"name": "...", "arguments": {...}}, ...]
    }

GET /health
    Returns: {"status": "ok"}

How the response is split
--------------------------
The model (xLAM-2-3b-fc-r) produces either:
  • A raw JSON array  → tool calls, e.g. [{"name": "set_light_state", "arguments": {...}}]
  • Plain text        → natural-language reply
  • Mixed             → text preamble + JSON array  (rare but handled)

Parsing rules:
  1. Search the full response for the FIRST '[' that starts a valid JSON array.
  2. Everything before that '[' is `thinking`.
  3. The JSON array is parsed into `tool_calls`.
  4. Everything after the JSON array is appended to `expression`.
  5. If no JSON array is found, the whole response is `expression`.
"""

import json
import re
import importlib.util
import os
import time
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load system prompt + tools from system.py (same directory)
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "system", os.path.join(os.path.dirname(__file__), "system.py")
)
_system_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_system_mod)

SYSTEM_PROMPT: str = _system_mod.SYSTEM_PROMPT
TOOLS: list = _system_mod.TOOLS

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
log.info("Loading model…")
_t0 = time.perf_counter()

llm = Llama.from_pretrained(
    repo_id="Salesforce/xLAM-2-3b-fc-r-gguf",
    filename="xLAM-2-3B-fc-r-Q8_0.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False,
)

log.info("Model loaded in %.3fs", time.perf_counter() - _t0)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="LLM Server", version="1.0.0")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    prompt: str
    stream: bool = False


class ParsedResponse(BaseModel):
    thinking: str = ""
    expression: str = ""
    tool_calls: list = []
    inference_time_s: float = 0.0
    tokens: int = 0


# ---------------------------------------------------------------------------
# Helper: parse raw LLM text into thinking / expression / tool_calls
# ---------------------------------------------------------------------------
def parse_response(raw: str) -> dict:
    """
    Split *raw* LLM output into thinking, expression, and tool_calls.

    Strategy
    --------
    1. Look for the first '[' that begins a valid JSON array anywhere in *raw*.
    2. Walk forward from that position to find the matching ']' (handles nesting).
    3. Everything before the '[' → thinking (stripped).
    4. Parse the JSON array → tool_calls.
    5. Everything after the closing ']' → expression (stripped).
    6. If no valid JSON array is found → expression = raw, others empty.
    """
    thinking = ""
    expression = ""
    tool_calls = []

    # Find candidate positions for a JSON array start
    for match in re.finditer(r'\[', raw):
        start = match.start()
        # Try to find the matching closing bracket using a simple depth counter
        depth = 0
        end = -1
        for i, ch in enumerate(raw[start:], start=start):
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end == -1:
            continue  # unmatched bracket, try next

        candidate = raw[start:end]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue  # not valid JSON, try next

        # Must be a list of dicts with at least a "name" key to count as tool calls
        if (
            isinstance(parsed, list)
            and len(parsed) > 0
            and all(isinstance(item, dict) and "name" in item for item in parsed)
        ):
            thinking = raw[:start].strip()
            tool_calls = parsed
            expression = raw[end:].strip()
            return {"thinking": thinking, "expression": expression, "tool_calls": tool_calls}

    # No tool-call JSON found → plain text response
    expression = raw.strip()
    return {"thinking": thinking, "expression": expression, "tool_calls": tool_calls}


# ---------------------------------------------------------------------------
# Helper: run inference and collect full response text
# ---------------------------------------------------------------------------
def run_inference(prompt: str) -> tuple[str, float, int]:
    """Returns (raw_text, inference_time_seconds, token_count)."""
    t0 = time.perf_counter()

    stream = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        tools=TOOLS,
        stream=True,
    )

    parts = []
    token_count = 0
    for chunk in stream:
        delta = chunk["choices"][0]["delta"]
        content = delta.get("content", "")
        if content:
            parts.append(content)
            token_count += 1

    elapsed = time.perf_counter() - t0
    raw = "".join(parts)
    return raw, elapsed, token_count


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ParsedResponse)
def chat(req: ChatRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    log.info("Prompt: %s", req.prompt)

    if req.stream:
        # Streaming: yield server-sent events with incremental tokens,
        # then a final JSON summary event.
        def event_generator():
            t0 = time.perf_counter()
            stream = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": req.prompt},
                ],
                tools=TOOLS,
                stream=True,
            )
            parts = []
            token_count = 0
            for chunk in stream:
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content", "")
                if content:
                    parts.append(content)
                    token_count += 1
                    yield f"data: {json.dumps({'token': content})}\n\n"

            elapsed = time.perf_counter() - t0
            raw = "".join(parts)
            parsed = parse_response(raw)
            parsed["inference_time_s"] = round(elapsed, 3)
            parsed["tokens"] = token_count
            yield f"data: {json.dumps({'result': parsed})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # Non-streaming
    raw, elapsed, tokens = run_inference(req.prompt)
    log.info("Raw response: %s", raw)
    log.info("Inference: %.3fs, %d tokens", elapsed, tokens)

    parsed = parse_response(raw)
    return ParsedResponse(
        thinking=parsed["thinking"],
        expression=parsed["expression"],
        tool_calls=parsed["tool_calls"],
        inference_time_s=round(elapsed, 3),
        tokens=tokens,
    )


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("llm_server:app", host="0.0.0.0", port=8000, reload=False)
