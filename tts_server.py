from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from kokoro import KPipeline
import soundfile as sf
import os
import uuid
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import contextlib
from contextlib import asynccontextmanager
import shutil
import traceback
import logging
import torch
import numpy as np
import uvicorn
import socket
import argparse
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

class PipelineManager:
    def __init__(self):
        self.pipeline = None

    def initialize(self):
        self.pipeline = KPipeline(lang_code='a')

    def cleanup(self):
        self.pipeline = None

    def get_pipeline(self):
        return self.pipeline

# Create pipeline manager instance
pipeline_manager = PipelineManager()

class TextRequest(BaseModel):
    text: str
    voice: str = Field(default='hf_alpha', description="Voice model to use for synthesis")
    speed: float = Field(default=1.0, ge=0.1, le=2.0, description="Speech speed multiplier")
    lang_code: str = Field(default='a', description="Language code ('a' for American English)")

class SegmentResponse(BaseModel):
    segment_id: int
    text_segment: str
    phonemes: str
    filepath: str

def combine_audio_segments(audio_segments):
    """Safely combine audio segments with error handling"""
    if not audio_segments:
        raise ValueError("No audio segments to combine")
    if len(audio_segments) == 1:
        return audio_segments[0]
    try:
        # Handle different types of audio segments
        if isinstance(audio_segments[0], torch.Tensor):
            # For PyTorch tensors
            return torch.cat(audio_segments, dim=0)
        elif isinstance(audio_segments[0], np.ndarray):
            # For numpy arrays
            return np.concatenate(audio_segments)
        else:
            # For other types that might have concatenate method
            return audio_segments[0].__class__.concatenate(audio_segments)
    except Exception as e:
        logger.error(f"Error combining audio segments: {str(e)}")
        raise ValueError(f"Failed to combine audio segments: {str(e)}")

def cleanup_file(filepath: str):
    """Safely cleanup a file"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        logger.error(f"Error cleaning up file {filepath}: {str(e)}")

@contextlib.contextmanager
def temporary_wav_file():
    """Context manager for handling temporary WAV files"""
    temp_filepath = os.path.join(AUDIO_DIR, f"temp_{uuid.uuid4()}.wav")
    try:
        yield temp_filepath
    finally:
        cleanup_file(temp_filepath)

def create_temp_copy(filepath):
    """Create a temporary copy of a file that will be served"""
    temp_filepath = os.path.join(AUDIO_DIR, f"serve_{uuid.uuid4()}.wav")
    shutil.copy2(filepath, temp_filepath)
    return temp_filepath

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize on startup
    pipeline_manager.initialize()
    yield
    # Cleanup on shutdown
    pipeline_manager.cleanup()

app = FastAPI(title="Kokoro TTS API", lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def generate_audio(
    request: TextRequest,
    return_segments: bool = Query(False, description="Return detailed segment information")
):
    """
    Generate audio from text and return file information.
    If return_segments=True, returns detailed information about each segment.
    Otherwise, returns filepath of the combined audio file.
    """
    pipeline = pipeline_manager.get_pipeline()
    if not pipeline:
        raise HTTPException(status_code=503, detail="TTS pipeline not initialized")

    segments = []
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")

        # Limit text length
        if len(request.text) > 10000:  # Adjust this limit as needed
            raise HTTPException(status_code=400, detail="Text too long")

        # Log the request parameters
        logger.debug(f"Processing text of length {len(request.text)} with voice {request.voice} and speed {request.speed}")

        generator = pipeline(
            request.text.strip(),  # Strip whitespace
            voice=request.voice,
            speed=request.speed
        )
        
        all_audio = []
        
        for i, (text_segment, phonemes, audio) in enumerate(generator):
            logger.debug(f"Processing segment {i}: {text_segment[:50]}...")
            all_audio.append(audio)
            if return_segments:
                # Create segment file
                filename = f"{uuid.uuid4()}_seg{i}.wav"
                filepath = os.path.join(AUDIO_DIR, filename)
                # Convert to numpy if tensor
                if isinstance(audio, torch.Tensor):
                    audio_data = audio.cpu().numpy()
                else:
                    audio_data = audio
                sf.write(filepath, audio_data, 24000)
                
                segments.append(SegmentResponse(
                    segment_id=i,
                    text_segment=text_segment,
                    phonemes=phonemes,
                    filepath=filepath
                ))
        
        try:
            combined_audio = combine_audio_segments(all_audio)
            # Convert to numpy if tensor
            if isinstance(combined_audio, torch.Tensor):
                combined_audio = combined_audio.cpu().numpy()
        except ValueError as e:
            logger.error(f"Error combining audio segments: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        
        # Create combined file
        filename = f"{uuid.uuid4()}.wav"
        filepath = os.path.join(AUDIO_DIR, filename)
        sf.write(filepath, combined_audio, 24000)
        
        if return_segments:
            return segments
        else:
            return {"filepath": filepath, "filename": os.path.basename(filepath)}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in generate_audio: {str(e)}\n{traceback.format_exc()}")
        # Clean up any created segment files if there's an error
        if segments:
            for segment in segments:
                cleanup_file(segment.filepath)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/{filename}")
async def get_audio(filename: str, background_tasks: BackgroundTasks):
    """Retrieve a generated audio file by filename"""
    filepath = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    # Create a temporary copy for serving
    try:
        temp_filepath = create_temp_copy(filepath)
        background_tasks.add_task(cleanup_file, temp_filepath)
        return FileResponse(
            temp_filepath,
            media_type="audio/wav"
        )
    except Exception as e:
        logger.error(f"Error serving audio file: {str(e)}")
        cleanup_file(temp_filepath)
        raise HTTPException(status_code=500, detail="Error serving audio file")

@app.post("/generate_direct")
async def generate_direct(request: TextRequest, background_tasks: BackgroundTasks):
    """
    Generate audio from text and return it directly in the response.
    Returns combined audio from all segments.
    """
    pipeline = pipeline_manager.get_pipeline()
    if not pipeline:
        raise HTTPException(status_code=503, detail="TTS pipeline not initialized")

    temp_filepath = None
    serve_filepath = None
    
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")

        # Limit text length
        if len(request.text) > 5000:  # Smaller limit for direct generation
            raise HTTPException(status_code=400, detail="Text too long for direct generation")

        logger.debug(f"Direct generation for text of length {len(request.text)}")

        generator = pipeline(
            request.text.strip(),  # Strip whitespace
            voice=request.voice,
            speed=request.speed
        )
        
        all_audio = []
        for i, (text_segment, _, audio) in enumerate(generator):
            logger.debug(f"Processing direct segment {i}: {text_segment[:50]}...")
            all_audio.append(audio)
        
        try:
            combined_audio = combine_audio_segments(all_audio)
            # Convert to numpy if tensor
            if isinstance(combined_audio, torch.Tensor):
                combined_audio = combined_audio.cpu().numpy()
        except ValueError as e:
            logger.error(f"Error combining audio segments in direct generation: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        
        # Create temporary files
        temp_filepath = os.path.join(AUDIO_DIR, f"direct_{uuid.uuid4()}.wav")
        sf.write(temp_filepath, combined_audio, 24000)
        
        serve_filepath = create_temp_copy(temp_filepath)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_file, temp_filepath)
        background_tasks.add_task(cleanup_file, serve_filepath)
        
        return FileResponse(
            serve_filepath,
            media_type="audio/wav",
            filename="generated_audio.wav"
        )
            
    except HTTPException:
        if temp_filepath:
            cleanup_file(temp_filepath)
        if serve_filepath:
            cleanup_file(serve_filepath)
        raise
    except Exception as e:
        logger.error(f"Error in generate_direct: {str(e)}\n{traceback.format_exc()}")
        if temp_filepath:
            cleanup_file(temp_filepath)
        if serve_filepath:
            cleanup_file(serve_filepath)
        raise HTTPException(status_code=500, detail=str(e))


def find_free_port(start_port=8000, max_attempts=100):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None

def main():
    parser = argparse.ArgumentParser(description='Run Kokoro API server')
    parser.add_argument('--port', type=int, default=8001, help='Port to run the server on')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host to run the server on')
    parser.add_argument('--auto-port', action='store_true', help='Automatically find an available port')
    args = parser.parse_args()

    port = args.port
    if args.auto_port or port == 8001:
        available_port = find_free_port(start_port=port)
        if available_port is None:
            print(f"Error: Could not find an available port starting from {port}")
            sys.exit(1)
        if available_port != port:
            print(f"Port {port} is in use, using port {available_port} instead")
            port = available_port

    print(f"Starting server on http://{args.host}:{port}")
    print(f"Documentation available at:")
    print(f"  - http://{args.host}:{port}/docs")
    print(f"  - http://{args.host}:{port}/redoc")

    uvicorn.run(
        "tts_server:app",
        host=args.host,
        port=port,
        reload=True
    )

if __name__ == "__main__":
    main()

