"""
agent_server.py — Voice agent orchestrator.

Ties together the three standalone servers:
  • stt_server   (port 5002) — Canary speech-to-text
  • llm_server   (port 8000) — xLAM-2-3b function-calling LLM
  • tts_server   (port 8001) — Kokoro text-to-speech

Flow
----
1. Client POSTs audio  →  /converse
2. Agent POSTs audio   →  stt_server /transcribe          → transcription
3. Agent POSTs text    →  llm_server /chat                → expression + tool_calls
4. For each tool call:
     • "talk"          →  tts_server /generate_direct     → WAV bytes, returned to client
     • other tools     →  logged + acknowledged (stub)
5. If LLM returns a plain expression with no talk call,
   the expression is synthesised via TTS and returned to the client.
6. All intermediate events are streamed as NDJSON lines so the client
   can display progress in real time.

Endpoints
---------
POST /converse
    Body : multipart/form-data, field "audio" — WAV/PCM audio file.
    Returns : streaming NDJSON (application/x-ndjson).
    Events:
        {"event": "transcription", "text": "...", "elapsed_s": ...}
        {"event": "llm",           "expression": "...", "tool_calls": [...],
                                   "elapsed_s": ...}
        {"event": "tool_call",     "name": "...", "arguments": {...}}
        {"event": "audio",         "data": "<base64-encoded WAV>",
                                   "mime": "audio/wav"}
        {"event": "done",          "total_elapsed_s": ...}
        {"event": "error",         "error": "..."}

GET /health
    Returns JSON: {"status": "ok", "backends": {...}}

Run
---
    python agent_server.py [--port 5010]
"""

import argparse
import base64
import json
import logging
import os
import threading
import time

import requests
from flask import Flask, Response, jsonify, request, stream_with_context

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STT_URL     = os.environ.get("STT_URL",  "http://127.0.0.1:8002")
LLM_URL     = os.environ.get("LLM_URL",  "http://127.0.0.1:8000")
TTS_URL     = os.environ.get("TTS_URL",  "http://127.0.0.1:8001")
DEFAULT_PORT = int(os.environ.get("AGENT_PORT", "8003"))

DEFAULT_TTS_VOICE = "af_heart"
REQUEST_TIMEOUT   = 120   # seconds for individual backend calls

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s [agent] %(levelname)s %(message)s",
)
log = logging.getLogger("agent_server")
log.setLevel(_log_level)

# ---------------------------------------------------------------------------
# Serialise all converse requests — prevents two simultaneous TTS playbacks
# ---------------------------------------------------------------------------
_converse_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

_session = requests.Session()
# Disable automatic retries — we never want a POST replayed silently.
_session.mount("http://",  requests.adapters.HTTPAdapter(max_retries=0))
_session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))


def _stt_transcribe(audio_bytes: bytes, filename: str) -> tuple[str, float]:
    """POST audio to stt_server and return (transcription, elapsed_s)."""
    log.debug("STT request: %d bytes, filename=%r", len(audio_bytes), filename)
    resp = _session.post(
        f"{STT_URL}/transcribe",
        files={"audio": (filename, audio_bytes, "audio/wav")},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    transcription = data.get("transcription", "").strip()
    elapsed = data.get("elapsed_s", 0.0)
    log.info("STT: %.3fs | %r", elapsed, transcription[:80])
    return transcription, elapsed


def _llm_chat(prompt: str) -> tuple[str, list, float]:
    """POST prompt to llm_server and return (expression, tool_calls, elapsed_s)."""
    log.debug("LLM request: %r", prompt[:120])
    t0 = time.perf_counter()
    resp = _session.post(
        f"{LLM_URL}/chat",
        json={"prompt": prompt, "stream": False},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    elapsed = data.get("inference_time_s", time.perf_counter() - t0)
    expression  = data.get("expression",  "").strip()
    tool_calls  = data.get("tool_calls",  [])
    log.info(
        "LLM: %.3fs | expression=%r tool_calls=%d",
        elapsed, expression[:80], len(tool_calls),
    )
    return expression, tool_calls, elapsed


def _tts_synthesise(text: str, voice: str = DEFAULT_TTS_VOICE) -> bytes:
    """POST text to tts_server /generate_direct and return raw WAV bytes."""
    log.debug("TTS request: voice=%r text=%r", voice, text[:80])
    t0 = time.perf_counter()
    resp = _session.post(
        f"{TTS_URL}/generate_direct",
        json={"text": text, "voice": voice, "speed": 1.0, "lang_code": "a"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    wav_bytes = resp.content
    log.info("TTS: %.3fs | %d bytes", time.perf_counter() - t0, len(wav_bytes))
    return wav_bytes


def _dispatch_tool(name: str, arguments: dict) -> dict:
    """
    Execute a non-talk tool call.  Currently a stub that logs and acknowledges.
    Extend this function to integrate real smart-home backends.
    Returns a result dict.
    """
    log.info("Tool dispatch: %s(%r)", name, arguments)
    # TODO: route to real home-automation backends
    return {"status": "ok", "tool": name, "arguments": arguments}


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    """Check reachability of all three backend servers."""
    backends = {}
    for label, base_url in [("stt", STT_URL), ("llm", LLM_URL), ("tts", TTS_URL)]:
        try:
            r = _session.get(f"{base_url}/health", timeout=5)
            backends[label] = r.json() if r.ok else {"status": "error", "code": r.status_code}
        except Exception as exc:
            backends[label] = {"status": "unreachable", "error": str(exc)}
    overall = "ok" if all(b.get("status") == "ok" for b in backends.values()) else "degraded"
    return jsonify({"status": overall, "backends": backends})


@app.route("/converse", methods=["POST"])
def converse():
    """
    Full voice-agent pipeline: audio → STT → LLM → TTS → streamed response.

    The response is a stream of NDJSON lines.  Each line is a JSON object
    with an "event" field describing its type.  See module docstring for the
    full event schema.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No 'audio' field in request"}), 400
    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Read audio bytes now, before the generator runs (FileStorage is not
    # safe to access after the request context may have been torn down).
    audio_bytes = audio_file.read()
    filename    = audio_file.filename or "audio.wav"

    @stream_with_context
    def generate():
        if not _converse_lock.acquire(blocking=False):
            log.warning("Agent busy — rejecting duplicate request.")
            yield json.dumps({"event": "error", "error": "busy"}) + "\n"
            return

        t_total = time.perf_counter()
        try:
            # ── 1. Speech-to-text ─────────────────────────────────────────
            try:
                transcription, stt_elapsed = _stt_transcribe(audio_bytes, filename)
            except Exception as exc:
                log.exception("STT failed")
                yield json.dumps({"event": "error", "error": f"STT: {exc}"}) + "\n"
                return

            if not transcription:
                yield json.dumps({"event": "error", "error": "STT returned empty transcription"}) + "\n"
                return

            yield json.dumps({
                "event":     "transcription",
                "text":      transcription,
                "elapsed_s": round(stt_elapsed, 4),
            }) + "\n"

            # ── 2. LLM inference ──────────────────────────────────────────
            try:
                expression, tool_calls, llm_elapsed = _llm_chat(transcription)
            except Exception as exc:
                log.exception("LLM failed")
                yield json.dumps({"event": "error", "error": f"LLM: {exc}"}) + "\n"
                return

            yield json.dumps({
                "event":      "llm",
                "expression": expression,
                "tool_calls": tool_calls,
                "elapsed_s":  round(llm_elapsed, 4),
            }) + "\n"

            # ── 3. Tool dispatch ──────────────────────────────────────────
            spoke = False   # True once a "talk" tool has produced audio
            tts_voice = DEFAULT_TTS_VOICE

            for call in tool_calls:
                name      = call.get("name", "")
                arguments = call.get("arguments", {})

                yield json.dumps({
                    "event":     "tool_call",
                    "name":      name,
                    "arguments": arguments,
                }) + "\n"

                if name == "talk":
                    text_to_speak = arguments.get("text", "").strip()
                    voice         = arguments.get("voice", DEFAULT_TTS_VOICE)
                    if not text_to_speak:
                        log.warning("talk tool called with empty text — skipping TTS")
                        continue
                    try:
                        wav_bytes = _tts_synthesise(text_to_speak, voice)
                    except Exception as exc:
                        log.exception("TTS failed for talk tool")
                        yield json.dumps({"event": "error", "error": f"TTS: {exc}"}) + "\n"
                        continue

                    yield json.dumps({
                        "event": "audio",
                        "data":  base64.b64encode(wav_bytes).decode("ascii"),
                        "mime":  "audio/wav",
                        "text":  text_to_speak,
                    }) + "\n"
                    spoke = True

                else:
                    result = _dispatch_tool(name, arguments)
                    log.debug("Tool result for %r: %r", name, result)

            # ── 4. Fallback TTS if LLM gave expression but no talk call ───
            if not spoke and expression:
                log.info("No talk tool — synthesising expression via TTS.")
                try:
                    wav_bytes = _tts_synthesise(expression, tts_voice)
                    yield json.dumps({
                        "event": "audio",
                        "data":  base64.b64encode(wav_bytes).decode("ascii"),
                        "mime":  "audio/wav",
                        "text":  expression,
                    }) + "\n"
                except Exception as exc:
                    log.exception("Fallback TTS failed")
                    yield json.dumps({"event": "error", "error": f"TTS fallback: {exc}"}) + "\n"

            # ── 5. Done ───────────────────────────────────────────────────
            total_elapsed = time.perf_counter() - t_total
            log.info("Converse complete in %.3fs", total_elapsed)
            yield json.dumps({
                "event":           "done",
                "total_elapsed_s": round(total_elapsed, 4),
            }) + "\n"

        finally:
            _converse_lock.release()

    return Response(generate(), mimetype="application/x-ndjson")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice agent orchestrator server")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"Port to listen on (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument("--stt-url", default=STT_URL,  help=f"STT server base URL (default: {STT_URL})")
    parser.add_argument("--llm-url", default=LLM_URL,  help=f"LLM server base URL (default: {LLM_URL})")
    parser.add_argument("--tts-url", default=TTS_URL,  help=f"TTS server base URL (default: {TTS_URL})")
    args = parser.parse_args()

    # Allow CLI overrides to propagate
    STT_URL = args.stt_url
    LLM_URL = args.llm_url
    TTS_URL = args.tts_url

    log.info("STT → %s", STT_URL)
    log.info("LLM → %s", LLM_URL)
    log.info("TTS → %s", TTS_URL)
    log.info("Starting agent server on %s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port)
