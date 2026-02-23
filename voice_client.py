"""
voice_client.py — Microphone voice client for the agent server.

Captures audio from the microphone, uses Silero VAD to detect speech
segments, POSTs each segment to agent_server /converse, and plays back
the returned audio through the local speakers.

The server streams NDJSON events:
  • "transcription" — prints what was heard
  • "llm"           — prints the LLM expression / tool calls
  • "tool_call"     — prints each tool being executed
  • "audio"         — decodes base64 WAV and plays it via sounddevice
  • "done"          — prints timing summary
  • "error"         — prints the error and continues listening

Usage
-----
    python voice_client.py [--url URL] [--device DEVICE]
                           [--silence-ms MS] [--threshold FLOAT]
                           [--playback-device DEVICE]

Examples
--------
    python voice_client.py
    python voice_client.py --url http://192.168.1.10:8003/converse
    python voice_client.py --device 1 --silence-ms 800
"""

import argparse
import base64
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
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_URL        = "http://localhost:8003/converse"
SAMPLE_RATE        = 16_000   # Hz — required by Silero VAD
VAD_CHUNK          = 512      # samples per VAD inference (~32 ms at 16 kHz)
PRE_ROLL_CHUNKS    = 15       # chunks kept before speech onset (~480 ms)
DEFAULT_SILENCE_MS = 700      # ms of silence that ends a segment
DEFAULT_THRESHOLD  = 0.5      # VAD speech probability threshold

# ---------------------------------------------------------------------------
# HTTP session — no automatic retries so POSTs are never silently replayed
# ---------------------------------------------------------------------------
_session = requests.Session()
_session.mount("http://",  requests.adapters.HTTPAdapter(max_retries=0))
_session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_silero_vad():
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
    """Convert float32 numpy array (−1…1) to 16-bit PCM WAV bytes."""
    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def play_wav_bytes(wav_bytes: bytes, device=None) -> None:
    """Decode WAV bytes and play through sounddevice (blocking)."""
    buf = io.BytesIO(wav_bytes)
    data, samplerate = sf.read(buf, dtype="float32")
    # Ensure 2-D for sounddevice (samples, channels)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    sd.play(data, samplerate=samplerate, device=device)
    sd.wait()


def _fmt_tool_call(name: str, arguments: dict) -> str:
    try:
        args_str = json.dumps(arguments, ensure_ascii=False)
    except Exception:
        args_str = str(arguments)
    return f"[tool] {name}({args_str})"


# ---------------------------------------------------------------------------
# Network worker — runs in a background thread so mic capture is never stalled
# ---------------------------------------------------------------------------

def send_worker(send_q: queue.Queue, url: str, playback_device) -> None:
    """
    Consume (pcm_array, capture_time) items from *send_q*, POST each to the
    agent server, and handle the streaming NDJSON response.
    Sentinel value None causes the thread to exit.
    """
    while True:
        item = send_q.get()
        if item is None:
            break

        pcm, _t_captured = item
        wav_bytes = audio_to_wav_bytes(pcm)
        t_start   = time.perf_counter()

        try:
            with _session.post(
                url,
                files={"audio": ("segment.wav", wav_bytes, "audio/wav")},
                timeout=300,
                stream=True,
            ) as resp:
                if not resp.ok:
                    print(f"\n[HTTP {resp.status_code}] {resp.text}", flush=True)
                    send_q.task_done()
                    continue

                content_type = resp.headers.get("Content-Type", "")
                if "ndjson" not in content_type and "x-ndjson" not in content_type:
                    # Unexpected non-streaming response
                    try:
                        data = resp.json()
                        print(f"\n[Response] {data}", flush=True)
                    except Exception:
                        print(f"\n[Response] {resp.text}", flush=True)
                    send_q.task_done()
                    continue

                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"\n[Unparseable] {line!r}", flush=True)
                        continue

                    kind    = event.get("event", "")
                    elapsed = time.perf_counter() - t_start

                    if kind == "transcription":
                        text = event.get("text", "")
                        stt_s = event.get("elapsed_s", 0)
                        print(f"\n[{elapsed:.2f}s] 🎤 {text}  (stt={stt_s:.2f}s)", flush=True)

                    elif kind == "llm":
                        expr       = event.get("expression", "")
                        tool_calls = event.get("tool_calls", [])
                        llm_s      = event.get("elapsed_s", 0)
                        if expr:
                            print(f"[{elapsed:.2f}s] 🤖 {expr}  (llm={llm_s:.2f}s)", flush=True)
                        if tool_calls:
                            print(f"[{elapsed:.2f}s] 🔧 {len(tool_calls)} tool call(s)  (llm={llm_s:.2f}s)", flush=True)

                    elif kind == "tool_call":
                        name = event.get("name", "")
                        args = event.get("arguments", {})
                        print(f"[{elapsed:.2f}s] {_fmt_tool_call(name, args)}", flush=True)

                    elif kind == "audio":
                        text = event.get("text", "")
                        if text:
                            print(f"[{elapsed:.2f}s] 🔊 {text}", flush=True)
                        try:
                            wav = base64.b64decode(event["data"])
                            play_wav_bytes(wav, device=playback_device)
                        except Exception as exc:
                            print(f"\n[Playback error] {exc}", flush=True)

                    elif kind == "done":
                        total_s = event.get("total_elapsed_s", 0)
                        print(f"[{elapsed:.2f}s] ✓ done  (total={total_s:.2f}s)\n", flush=True)

                    elif kind == "error":
                        err = event.get("error", "")
                        if err == "busy":
                            # Server already handling a request; silently discard
                            pass
                        else:
                            print(f"\n[{elapsed:.2f}s] ⚠ {err}", flush=True)

                    else:
                        print(f"\n[unknown event] {line}", flush=True)

        except requests.RequestException as exc:
            print(f"\n[Request failed] {exc}", flush=True)

        send_q.task_done()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Voice client — mic → agent server → speaker"
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"Agent server /converse URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--device", default=None, type=int,
        help="Input (microphone) device index. Run `python -m sounddevice` to list.",
    )
    parser.add_argument(
        "--playback-device", default=None, type=int,
        dest="playback_device",
        help="Output (speaker) device index. Defaults to system default.",
    )
    parser.add_argument(
        "--silence-ms", default=DEFAULT_SILENCE_MS, type=int,
        help=f"Milliseconds of silence that ends a speech segment (default: {DEFAULT_SILENCE_MS})",
    )
    parser.add_argument(
        "--threshold", default=DEFAULT_THRESHOLD, type=float,
        help=f"VAD speech probability threshold 0–1 (default: {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()

    silence_chunks = max(1, int(args.silence_ms / 1000 * SAMPLE_RATE / VAD_CHUNK))

    # ------------------------------------------------------------------
    # Load VAD
    # ------------------------------------------------------------------
    vad_model = load_silero_vad()

    # ------------------------------------------------------------------
    # Audio capture queue (filled by sounddevice callback)
    # ------------------------------------------------------------------
    audio_q: queue.Queue = queue.Queue()

    def sd_callback(indata, frames, time_info, status):
        if status:
            print(f"[sounddevice] {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())   # keep mono float32

    # ------------------------------------------------------------------
    # Network send queue (consumed by background worker)
    # ------------------------------------------------------------------
    send_q: queue.Queue = queue.Queue()
    worker = threading.Thread(
        target=send_worker,
        args=(send_q, args.url, args.playback_device),
        daemon=True,
    )
    worker.start()

    # ------------------------------------------------------------------
    # Status banner
    # ------------------------------------------------------------------
    print(f"Agent URL      : {args.url}")
    print(f"Input device   : {args.device if args.device is not None else 'default'}")
    print(f"Output device  : {args.playback_device if args.playback_device is not None else 'default'}")
    print(f"Silence cutoff : {args.silence_ms} ms  ({silence_chunks} VAD chunks)")
    print(f"VAD threshold  : {args.threshold}")
    print("\nListening … (Ctrl-C to stop)\n", flush=True)

    # ------------------------------------------------------------------
    # VAD loop
    # ------------------------------------------------------------------
    pre_roll:       deque = deque(maxlen=PRE_ROLL_CHUNKS)
    speech_buffer:  list  = []
    in_speech             = False
    silence_counter       = 0
    leftover              = np.array([], dtype=np.float32)

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

                n_full   = len(raw) // VAD_CHUNK
                leftover = raw[n_full * VAD_CHUNK:]

                for i in range(n_full):
                    chunk  = raw[i * VAD_CHUNK:(i + 1) * VAD_CHUNK]
                    tensor = torch.from_numpy(chunk).unsqueeze(0)   # (1, 512)

                    with torch.no_grad():
                        prob = vad_model(tensor, SAMPLE_RATE).item()

                    is_speech = prob >= args.threshold

                    if is_speech:
                        if not in_speech:
                            in_speech       = True
                            silence_counter = 0
                            speech_buffer   = list(pre_roll)
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
                                speech_buffer   = []
                                silence_counter = 0
                        else:
                            pre_roll.append(chunk)

    except KeyboardInterrupt:
        print("\n\nStopped.", flush=True)
    finally:
        # Flush any in-progress speech segment
        if speech_buffer:
            pcm = np.concatenate(speech_buffer)
            send_q.put((pcm, time.time()))
        send_q.put(None)   # sentinel — tells worker to exit
        send_q.join()


if __name__ == "__main__":
    main()
