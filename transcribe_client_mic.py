"""
Microphone transcription client — captures audio from the microphone,
uses Silero VAD to detect speech segments automatically, and sends each
segment to the server, printing results as they arrive.

By default the client posts to /transcribe_and_chat on the
Nanbeige+Canary server (port 5003), which transcribes the audio with
Canary and then passes the transcript to the Nanbeige4 LLM, returning
a streaming NDJSON response.  Tool calls are printed as soon as they
are processed by the server.  Point --url at /transcribe to get raw
transcriptions only (non-streaming JSON).

Usage
-----
    python transcribe_client_mic.py [--url URL] [--device DEVICE]
                                    [--silence-ms MS] [--threshold FLOAT]

Examples
--------
    python transcribe_client_mic.py
    python transcribe_client_mic.py --url http://localhost:5003/transcribe
    python transcribe_client_mic.py --device 1 --silence-ms 800
"""

import argparse
import io
import json
import queue
import sys
import threading
import time
import wave
from collections import deque

import numpy as np
import requests
import sounddevice as sd
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_URL        = "http://localhost:5003/transcribe_and_chat"
SAMPLE_RATE        = 16_000   # Hz — required by Silero VAD
VAD_CHUNK          = 512      # samples per VAD inference (~32 ms)
PRE_ROLL_CHUNKS    = 15       # chunks kept before speech starts (~480 ms)
DEFAULT_SILENCE_MS = 700      # ms of silence that ends a segment
DEFAULT_THRESHOLD  = 0.5      # VAD probability threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_silero_vad():
    """Download (or load cached) Silero VAD model."""
    print("Loading Silero VAD model …", flush=True)
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        verbose=False,
    )
    model.eval()
    print("Silero VAD ready.\n", flush=True)
    return model


def audio_to_wav_bytes(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert a float32 numpy array (−1…1) to WAV bytes (16-bit PCM)."""
    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def _format_tool_call(event: dict) -> str:
    """Return a human-readable string for a tool_call event."""
    name = event.get("name", "?")
    args = event.get("arguments", {})
    result_raw = event.get("result", "")
    # Pretty-print arguments
    try:
        args_str = json.dumps(args, ensure_ascii=False)
    except Exception:
        args_str = str(args)
    # Pretty-print result
    try:
        result_obj = json.loads(result_raw) if isinstance(result_raw, str) else result_raw
        result_str = json.dumps(result_obj, ensure_ascii=False)
    except Exception:
        result_str = str(result_raw)
    return f"[tool] {name}({args_str}) → {result_str}"


# One persistent session with retries disabled so urllib3 never silently
# resends a POST (which would cause the server to process the same audio twice).
_session = requests.Session()
_session.mount("http://",  requests.adapters.HTTPAdapter(max_retries=0))
_session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))


def send_audio_streaming(wav_bytes: bytes, url: str) -> None:
    """
    POST WAV bytes to the server using a streaming request and print events
    as they arrive (NDJSON format expected from /transcribe_and_chat).

    Each line from the server is a JSON object with an "event" field:
      • "tool_call"  — printed immediately when the tool result is ready
      • "response"   — the final LLM response, printed at the end
      • "error"      — server-side error, printed as a warning
    """
    t_start = time.perf_counter()
    try:
        with _session.post(
            url,
            files={"audio": ("segment.wav", wav_bytes, "audio/wav")},
            timeout=300,
            stream=True,
        ) as resp:
            if not resp.ok:
                print(f"\n[Error {resp.status_code}: {resp.text}]", flush=True)
                return

            content_type = resp.headers.get("Content-Type", "")

            # ── Streaming NDJSON path (application/x-ndjson) ──────────────
            if "ndjson" in content_type or "x-ndjson" in content_type:
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"\n[Unparseable server line: {line!r}]", flush=True)
                        continue

                    kind = event.get("event", "")

                    if kind == "tool_call":
                        elapsed = time.perf_counter() - t_start
                        print(f"\n[{elapsed:.2f}s] {_format_tool_call(event)}", flush=True)

                    elif kind == "response":
                        elapsed = time.perf_counter() - t_start
                        text = event.get("response", "").strip()
                        t_elapsed = event.get("transcribe_elapsed_s", 0)
                        l_elapsed = event.get("llm_elapsed_s", 0)
                        if text:
                            print(
                                f"\n[{elapsed:.2f}s | transcribe={t_elapsed:.2f}s"
                                f" llm={l_elapsed:.2f}s] {text}",
                                flush=True,
                            )

                    elif kind == "error":
                        elapsed = time.perf_counter() - t_start
                        err = event.get("error", "")
                        if err == "busy":
                            # Server is already handling a request — this is a
                            # duplicate that was correctly rejected; ignore it.
                            pass
                        else:
                            print(f"\n[{elapsed:.2f}s] [Server error] {err}", flush=True)

                    else:
                        # Unknown event — log it verbatim
                        print(f"\n[unknown event] {line}", flush=True)

            # ── Non-streaming JSON path (/transcribe or similar) ───────────
            else:
                data = resp.json()
                elapsed = time.perf_counter() - t_start
                text = (data.get("response") or data.get("transcription") or "").strip()
                if text:
                    print(f"\n[{elapsed:.2f}s] {text}", flush=True)

    except requests.RequestException as exc:
        print(f"\n[Request failed: {exc}]", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Microphone → Silero VAD → transcription/chat server"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Server endpoint URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=int,
        help="Input device index (default: system default). "
             "Run `python -m sounddevice` to list devices.",
    )
    parser.add_argument(
        "--silence-ms",
        default=DEFAULT_SILENCE_MS,
        type=int,
        help=f"Milliseconds of silence that ends a speech segment "
             f"(default: {DEFAULT_SILENCE_MS})",
    )
    parser.add_argument(
        "--threshold",
        default=DEFAULT_THRESHOLD,
        type=float,
        help=f"VAD speech probability threshold 0–1 (default: {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()

    silence_chunks = max(1, int(args.silence_ms / 1000 * SAMPLE_RATE / VAD_CHUNK))

    # ------------------------------------------------------------------
    # Load VAD
    # ------------------------------------------------------------------
    vad_model = load_silero_vad()

    # ------------------------------------------------------------------
    # Audio queue filled by the sounddevice callback
    # ------------------------------------------------------------------
    audio_q: queue.Queue = queue.Queue()

    def sd_callback(indata, frames, time_info, status):
        if status:
            print(f"[sounddevice] {status}", file=sys.stderr)
        # indata shape: (frames, channels) — keep mono, float32
        audio_q.put(indata[:, 0].copy())

    # ------------------------------------------------------------------
    # Network I/O happens in a background thread so mic capture is
    # never blocked waiting for the server.
    # ------------------------------------------------------------------
    send_q: queue.Queue = queue.Queue()

    def send_worker():
        while True:
            item = send_q.get()
            if item is None:   # sentinel → exit
                break
            pcm, _t_captured = item
            wav_bytes = audio_to_wav_bytes(pcm)
            send_audio_streaming(wav_bytes, args.url)
            send_q.task_done()

    worker = threading.Thread(target=send_worker, daemon=True)
    worker.start()

    # ------------------------------------------------------------------
    # VAD loop
    # ------------------------------------------------------------------
    print(f"Server   : {args.url}")
    print(f"Device   : {args.device if args.device is not None else 'default'}")
    print(f"Silence  : {args.silence_ms} ms  ({silence_chunks} VAD chunks)")
    print(f"Threshold: {args.threshold}")
    print("\nListening … (Ctrl-C to stop)\n", flush=True)

    pre_roll: deque = deque(maxlen=PRE_ROLL_CHUNKS)
    speech_buffer: list = []
    in_speech = False
    silence_counter = 0
    leftover = np.array([], dtype=np.float32)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=VAD_CHUNK,
            device=args.device,
            callback=sd_callback,
        ):
            while True:
                raw = audio_q.get()
                if leftover.size:
                    raw = np.concatenate([leftover, raw])

                n_full = len(raw) // VAD_CHUNK
                leftover = raw[n_full * VAD_CHUNK:]

                for i in range(n_full):
                    chunk = raw[i * VAD_CHUNK:(i + 1) * VAD_CHUNK]
                    tensor = torch.from_numpy(chunk).unsqueeze(0)   # (1, 512)

                    with torch.no_grad():
                        prob = vad_model(tensor, SAMPLE_RATE).item()

                    is_speech = prob >= args.threshold

                    if is_speech:
                        if not in_speech:
                            in_speech = True
                            silence_counter = 0
                            speech_buffer = list(pre_roll)
                            print("▶ ", end="", flush=True)
                        speech_buffer.append(chunk)
                        silence_counter = 0
                    else:
                        if in_speech:
                            speech_buffer.append(chunk)   # keep trailing silence
                            silence_counter += 1
                            if silence_counter >= silence_chunks:
                                in_speech = False
                                print("■", flush=True)
                                pcm = np.concatenate(speech_buffer)
                                send_q.put((pcm, time.time()))
                                speech_buffer = []
                                silence_counter = 0
                        else:
                            pre_roll.append(chunk)

    except KeyboardInterrupt:
        print("\n\nStopped.", flush=True)
    finally:
        if speech_buffer:
            pcm = np.concatenate(speech_buffer)
            send_q.put((pcm, time.time()))
        send_q.put(None)   # sentinel
        send_q.join()


if __name__ == "__main__":
    main()
