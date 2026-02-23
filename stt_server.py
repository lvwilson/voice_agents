"""
stt_server.py — Standalone Speech-to-Text server using Canary (nvidia/canary-qwen-2.5b).

Canary handles all speech-to-text transcription.  The server exposes a simple
HTTP API so that any client (e.g. transcribe_client_mic.py) can POST audio and
receive a transcription without needing the Nanbeige4 LLM to be loaded.

Endpoints
---------
POST /transcribe
    Transcribes audio and returns the Canary transcription.
    Body : multipart/form-data, field "audio" — WAV/PCM audio file.
    Returns JSON:
        {
            "transcription": "<text>",
            "backend":       "canary",
            "elapsed_s":     <float>
        }

GET /health
    Returns JSON: {"status": "ok", "backend": "canary"}

Run
---
    python stt_server.py [--port 5002] [--host 0.0.0.0]
"""

import sys
import os

# ── cuDNN path fix ────────────────────────────────────────────────────────────
_python_dir    = os.path.dirname(os.path.abspath(sys.executable))
_site_packages = os.path.join(_python_dir, "..", "lib", "python3.12", "site-packages")
_cudnn_lib     = os.path.normpath(os.path.join(_site_packages, "nvidia", "cudnn", "lib"))
if os.path.isdir(_cudnn_lib):
    _existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = _cudnn_lib + (":" + _existing if _existing else "")
    if os.environ.get("_CUDNN_PATH_SET") != "1":
        os.environ["_CUDNN_PATH_SET"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

import time
import tempfile
import argparse
import logging
import threading

import numpy as np
import soundfile as sf
import torch
import torchaudio
from flask import Flask, request, jsonify
from nemo.collections.speechlm2.models import SALM

# ── Config ────────────────────────────────────────────────────────────────────
CANARY_MODEL_NAME = "nvidia/canary-qwen-2.5b"
CANARY_MAX_TOKENS = 128
DEFAULT_PORT      = 8002

# ── Logging ───────────────────────────────────────────────────────────────────
_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "DEBUG").upper(), logging.DEBUG)
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s [stt-server] %(levelname)s %(message)s",
)
log = logging.getLogger("stt_server")
log.setLevel(_log_level)

# ── CUDA check ────────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This server requires a CUDA-capable GPU.")

device = torch.device("cuda")
log.info("CUDA device: %s", torch.cuda.get_device_name(0))

# ── Load Canary ASR model ─────────────────────────────────────────────────────
log.info("Loading Canary ASR model (%s) onto CUDA in float16…", CANARY_MODEL_NAME)
t0 = time.perf_counter()
canary_model = SALM.from_pretrained(CANARY_MODEL_NAME)
canary_model = canary_model.to(device=device, dtype=torch.float16)
canary_model.eval()
torch.cuda.synchronize()
log.info("Canary model loaded in %.2fs", time.perf_counter() - t0)


# ── Patch embed_tokens if missing (transformers compatibility) ────────────────
def _patch_embed_tokens(model):
    """
    Ensure the inner LLM model has an embed_tokens attribute.
    Newer Qwen3 architectures may expose it under a different name.
    No-op if already present.
    """
    _CANDIDATES = ("tok_embeddings", "wte", "word_embeddings", "embedding")
    inner = None
    for attr_path in ("llm.model", "llm", "model"):
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            inner = obj
            break
        except AttributeError:
            continue

    if inner is None:
        log.warning("_patch_embed_tokens: could not locate inner LLM model object.")
        return

    if hasattr(inner, "embed_tokens"):
        log.debug("_patch_embed_tokens: embed_tokens already present — no patch needed.")
        return

    for cand in _CANDIDATES:
        if hasattr(inner, cand):
            log.warning(
                "_patch_embed_tokens: patching embed_tokens → %s on %s",
                cand, type(inner).__name__,
            )
            inner.embed_tokens = getattr(inner, cand)
            return

    log.warning(
        "_patch_embed_tokens: could not find a suitable embedding attribute on %s; "
        "generation may fail with AttributeError.",
        type(inner).__name__,
    )


_patch_embed_tokens(canary_model)

# ── Inference lock — serialize requests so the GPU is never shared ────────────
_inference_lock = threading.Lock()


# ── Helper: transcribe audio with Canary ─────────────────────────────────────
def canary_transcribe(audio_file) -> tuple[str, float]:
    """
    Save *audio_file* (a Flask FileStorage object) to disk, resample to
    16 kHz mono WAV, run Canary transcription, and return
    (transcription_text, elapsed_seconds).
    """
    suffix = os.path.splitext(audio_file.filename)[1] or ".wav"
    log.debug("Transcription request: filename=%r suffix=%r", audio_file.filename, suffix)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        upload_path = tmp.name
        audio_file.save(upload_path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    try:
        # ── Load & normalise audio ────────────────────────────────────────
        audio_np, sample_rate = sf.read(upload_path, always_2d=True)
        n_channels   = audio_np.shape[1]
        n_samples_in = audio_np.shape[0]
        duration_s   = n_samples_in / sample_rate
        log.debug(
            "Audio loaded: sample_rate=%d Hz  channels=%d  samples=%d  duration=%.3fs",
            sample_rate, n_channels, n_samples_in, duration_s,
        )

        # Stereo → mono
        audio_np = audio_np.mean(axis=1)
        waveform = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0)

        # Resample to 16 kHz if needed
        if sample_rate != 16000:
            log.debug("Resampling audio from %d Hz → 16000 Hz", sample_rate)
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        n_samples_out = waveform.shape[-1]
        log.debug(
            "Audio after resample: samples=%d  duration=%.3fs",
            n_samples_out, n_samples_out / 16000,
        )

        sf.write(wav_path, waveform.squeeze(0).numpy(), 16000, format="WAV", subtype="PCM_16")

        # ── Build Canary prompt ───────────────────────────────────────────
        prompt = [
            {
                "role": "user",
                "content": f"Transcribe the following: {canary_model.audio_locator_tag}",
                "audio": [wav_path],
            }
        ]
        log.debug("Canary prompt content: %r", prompt[0]["content"])

        # ── Run inference ─────────────────────────────────────────────────
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                answer_ids = canary_model.generate(
                    prompts=[prompt],
                    max_new_tokens=CANARY_MAX_TOKENS,
                )
        except AttributeError as _ae:
            if "embed_tokens" not in str(_ae):
                raise
            log.warning(
                "Caught AttributeError '%s' during Canary generate(); "
                "attempting runtime embed_tokens patch and retrying…",
                _ae,
            )
            _patch_embed_tokens(canary_model)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                answer_ids = canary_model.generate(
                    prompts=[prompt],
                    max_new_tokens=CANARY_MAX_TOKENS,
                )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start

        n_output_tokens = len(answer_ids[0])
        tok_per_sec = n_output_tokens / elapsed if elapsed > 0 else 0.0
        log.debug(
            "Canary generation complete: %d output tokens in %.3fs (%.1f tok/s)",
            n_output_tokens, elapsed, tok_per_sec,
        )
        log.debug("Canary raw token IDs: %s", answer_ids[0].cpu().tolist())

        transcription = canary_model.tokenizer.ids_to_text(answer_ids[0].cpu())
        log.debug("Canary transcription: %r", transcription)

        log.info(
            "Transcription | duration=%.3fs  output_tokens=%d  elapsed=%.3fs  tok/s=%.1f",
            duration_s, n_output_tokens, elapsed, tok_per_sec,
        )

    finally:
        os.unlink(upload_path)
        os.unlink(wav_path)

    return transcription, elapsed


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "backend": "canary"})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Transcribe uploaded audio using Canary STT.

    Expects multipart/form-data with an "audio" field containing a WAV/PCM
    audio file.  Returns JSON with the transcription text.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No 'audio' field in request"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    with _inference_lock:
        try:
            transcription, elapsed = canary_transcribe(audio_file)
        except Exception as exc:
            log.exception("Transcription failed")
            return jsonify({"error": str(exc)}), 500

    log.info("Transcribed in %.3fs | %.80s…", elapsed, transcription)
    log.debug("/transcribe full transcription:\n%s", transcription)

    return jsonify({
        "transcription": transcription,
        "backend":       "canary",
        "elapsed_s":     round(elapsed, 4),
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canary STT server")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"Port to listen on (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    log.info("Starting Canary STT server on %s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port)
