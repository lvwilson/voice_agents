# Voice Agent

A local, GPU-accelerated voice assistant for smart-home control.  Speak into
your microphone; the system transcribes your speech, reasons about it with a
function-calling LLM, executes home-automation tool calls, and speaks a
response back through your speakers — all running on your own hardware.

```
Microphone → [voice_client] → [agent_server :8003]
                                      │
                  ┌───────────────────┼───────────────────┐
                  ▼                   ▼                   ▼
          [stt_server :8002]  [llm_server :8000]  [tts_server :8001]
          Canary (NeMo)       xLAM-2-3b (llama)   Kokoro TTS
```

---

## Servers

| Server | Port | Description |
|---|---|---|
| `llm_server.py` | **8000** | xLAM-2-3b-fc-r function-calling LLM via llama-cpp |
| `tts_server.py` | **8001** | Kokoro text-to-speech |
| `stt_server.py` | **8002** | Canary (nvidia/canary-qwen-2.5b) speech-to-text |
| `agent_server.py` | **8003** | Orchestrator — ties the three backends together |

All ports are in the **8000–8010** range and can be overridden with environment
variables (`LLM_URL`, `TTS_URL`, `STT_URL`, `AGENT_PORT`) or CLI flags.

---

## Requirements

- Python 3.12
- CUDA-capable GPU (required by `stt_server` and recommended for `llm_server`)
- The following Python packages (install into your environment):

```
flask
fastapi
uvicorn
requests
pydantic
llama-cpp-python        # llm_server
nemo_toolkit[speechlm2] # stt_server  (nvidia/canary-qwen-2.5b)
kokoro                  # tts_server
soundfile
sounddevice
torchaudio
numpy
```

> **Note:** `llama-cpp-python` should be built with CUDA support for GPU
> offloading (`CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python`).

---

## Quick Start

### 1. Start all servers

```bash
./start_servers.sh
```

This launches all four servers in the background, writes logs to `./logs/`,
and waits for each backend to pass its `/health` check before printing a
status summary.

```
=== Starting servers ===
  [llm_server]   starting on port 8000 …  pid=12345  log=logs/llm_server.log
  [tts_server]   starting on port 8001 …  pid=12346  log=logs/tts_server.log
  [stt_server]   starting on port 8002 …  pid=12347  log=logs/stt_server.log
  [agent_server] starting on port 8003 …  pid=12348  log=logs/agent_server.log

=== Waiting for backends to become healthy ===
  [llm_server]   healthy after 4s
  [tts_server]   healthy after 6s
  [stt_server]   healthy after 18s
```

### 2. Run the voice client

```bash
python voice_client.py
```

Speak naturally. The client uses **Silero VAD** to detect speech segments,
sends each one to the agent, and plays the spoken response through your
speakers.

---

## start_servers.sh

```
Usage: ./start_servers.sh {start|stop|status|restart}
```

| Command | Effect |
|---|---|
| `start` | Start all servers (default) |
| `stop` | Gracefully stop all servers (SIGTERM → SIGKILL after 10 s) |
| `status` | Show running/stopped state and PIDs |
| `restart` | Stop then start |

Logs: `logs/<server>.log`  
PIDs: `logs/<server>.pid`

---

## Conversation Flow

```
1. voice_client  captures mic audio via sounddevice + Silero VAD
2. voice_client  POSTs WAV segment  →  agent_server /converse
3. agent_server  POSTs audio        →  stt_server /transcribe
                                        ← transcription text
4. agent_server  POSTs text         →  llm_server /chat
                                        ← expression + tool_calls JSON
5. agent_server  dispatches tools:
     • "talk"   →  tts_server /generate_direct  →  WAV bytes
     • others   →  home-automation stubs (logged)
6. agent_server  streams NDJSON events back to voice_client
7. voice_client  prints events and plays audio via sounddevice
```

The agent server streams **NDJSON** so the client can display progress in real
time as each stage completes.

---

## API Reference

### agent_server — `POST /converse`

Accepts a `multipart/form-data` request with an `audio` field (WAV/PCM).
Returns a streaming `application/x-ndjson` response.

**Events:**

```jsonc
{"event": "transcription", "text": "turn on the living room lights", "elapsed_s": 1.2}
{"event": "llm",           "expression": "", "tool_calls": [...], "elapsed_s": 0.8}
{"event": "tool_call",     "name": "set_light_power", "arguments": {"room": "living room", "state": "on"}}
{"event": "tool_call",     "name": "talk", "arguments": {"text": "Turning on the living room lights."}}
{"event": "audio",         "data": "<base64 WAV>", "mime": "audio/wav", "text": "Turning on the living room lights."}
{"event": "done",          "total_elapsed_s": 3.1}
// on any failure:
{"event": "error",         "error": "STT: connection refused"}
```

### agent_server — `GET /health`

```json
{
  "status": "ok",
  "backends": {
    "stt": {"status": "ok", "backend": "canary"},
    "llm": {"status": "ok"},
    "tts": {"status": "ok"}
  }
}
```

### llm_server — `POST /chat`

```json
// Request
{"prompt": "turn off all the lights", "stream": false}

// Response
{
  "thinking":        "",
  "expression":      "",
  "tool_calls":      [{"name": "set_light_power", "arguments": {"room": "all", "state": "off"}},
                      {"name": "talk",             "arguments": {"text": "All lights off."}}],
  "inference_time_s": 0.74,
  "tokens":          42
}
```

Streaming mode (`"stream": true`) returns SSE tokens followed by a final
`{"result": {...}}` event.

### stt_server — `POST /transcribe`

```
multipart/form-data  field: audio  (WAV/PCM file)
```
```json
{"transcription": "turn off all the lights", "backend": "canary", "elapsed_s": 1.18}
```

### tts_server — `POST /generate_direct`

```json
// Request
{"text": "All lights off.", "voice": "af_heart", "speed": 1.0, "lang_code": "a"}

// Response: raw WAV bytes (audio/wav)
```

---

## Available Tools

The LLM can call the following tools (defined in `system.py`):

| Tool | Description |
|---|---|
| `set_light_power` | Turn lights on/off in a room |
| `set_light_color` | Change light colour and brightness |
| `play_music` | Play, pause, skip, or stop music |
| `water_garden` | Start/stop garden watering zones |
| `set_ac_temperature` | Set AC/heating temperature or mode |
| `talk` | Speak text aloud via TTS |

Tool calls other than `talk` are currently **stubs** — they are logged and
acknowledged. Extend `_dispatch_tool()` in `agent_server.py` to wire them up
to real home-automation backends (e.g. Home Assistant, MQTT).

---

## Configuration

All settings can be overridden with environment variables:

| Variable | Default | Description |
|---|---|---|
| `STT_URL` | `http://127.0.0.1:8002` | STT server base URL |
| `LLM_URL` | `http://127.0.0.1:8000` | LLM server base URL |
| `TTS_URL` | `http://127.0.0.1:8001` | TTS server base URL |
| `AGENT_PORT` | `8003` | Agent server listen port |
| `LOG_LEVEL` | `INFO` | Python log level (`DEBUG`, `INFO`, `WARNING`, …) |

Or pass `--stt-url`, `--llm-url`, `--tts-url`, `--port` flags directly to
`agent_server.py`.

---

## Models

| Component | Model |
|---|---|
| STT | [nvidia/canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b) — downloaded automatically by NeMo |
| LLM | [Salesforce/xLAM-2-3b-fc-r-gguf](https://huggingface.co/Salesforce/xLAM-2-3b-fc-r-gguf) (`xLAM-2-3B-fc-r-Q8_0.gguf`) — downloaded automatically by llama-cpp |
| TTS | [Kokoro](https://github.com/hexgrad/kokoro) — voice `af_heart` (American English) |

Models are downloaded on first run and cached by their respective frameworks.

---

## File Structure

```
.
├── agent_server.py          # Orchestrator (port 8003)
├── llm_server.py            # LLM inference server (port 8000)
├── stt_server.py            # Speech-to-text server (port 8002)
├── tts_server.py            # Text-to-speech server (port 8001)
├── voice_client.py          # Microphone client
├── system.py                # System prompt + tool definitions
├── start_servers.sh         # Start / stop / status / restart all servers
├── audio_files/             # Temporary TTS audio output (auto-created)
└── logs/                    # Server logs and PID files (auto-created)
```
