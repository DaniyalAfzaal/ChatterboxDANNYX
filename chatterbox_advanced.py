"""
Chatterbox TTS - Native Modal Job Queue
Fixed version with voices mount
"""
import modal
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import time
import uuid
import sys
import os

# --- Modal App Setup ---
app = modal.App("chatterbox-async-v1")
vol = modal.Volume.from_name("chatterbox-data", create_if_missing=True)

# Storage paths inside volume
OUTPUT_DIR = "/data/outputs"
VOICES_DIR = "/data/voices"
REF_AUDIO_DIR = "/data/reference_audio"
MODEL_DIR = "/data/model_cache"

# --- Docker Images ---
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "fastapi[standard]",
        "requests",
        "soundfile",
        "httpx",
        "pyyaml",
        "pydub",
        "numpy",
    )
)

worker_image = (
    base_image
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        "numpy<3.0.0",
        "soundfile",
        "librosa",
        "safetensors",
        "descript-audio-codec",
        "PyYAML",
        "python-multipart",
        "requests",
        "Jinja2",
        "watchdog",
        "aiofiles",
        "unidecode",
        "inflect",
        "tqdm",
        "hf_transfer",
        "pydub",
        "audiotsm",
        "praat-parselmouth",
        extra_index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install("git+https://github.com/devnen/chatterbox.git")
    .run_commands("rm -rf /app && git clone --branch main https://github.com/DaniyalAfzaal/ChatterboxDANNYX /app")
    .add_local_dir("voices", remote_path="/app/voices")
    .add_local_file("engine.py", "/app/engine.py")
    .add_local_file("config.py", "/app/config.py")
)

web_image = (
    base_image
    .run_commands("rm -rf /app && git clone --branch main https://github.com/DaniyalAfzaal/ChatterboxDANNYX /app")
    .add_local_dir("ui", remote_path="/app/ui")
)

def _split_text_by_words(text: str, max_words: int) -> list:
    """
    Split text into chunks at sentence boundaries, preserving ALL content.
    Uses improved sentence detection and extensive logging to ensure no data loss.
    """
    import re
    
    print(f"\n{'='*60}")
    print(f"CHUNKING STARTED")
    print(f"{'='*60}")
    print(f"Input text length: {len(text)} characters")
    print(f"Input word count: {len(text.split())} words")
    print(f"Max words per chunk: {max_words}")
    print(f"{'='*60}\n")
    
    # More robust sentence splitting - handles multiple punctuation patterns
    # Split on: period, exclamation, question mark - followed by space and capital letter OR end of string
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
    raw_sentences = re.split(sentence_pattern, text)
    
    # Clean and filter sentences
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if s:
            sentences.append(s)
            print(f"Sentence {len(sentences)}: {len(s.split())} words - \"{s[:50]}...\"" if len(s) > 50 else f"Sentence {len(sentences)}: {len(s.split())} words - \"{s}\"")
    
    print(f"\nTotal sentences detected: {len(sentences)}\n")
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for idx, sentence in enumerate(sentences, 1):
        sentence_words = len(sentence.split())
        print(f"Processing sentence {idx}/{len(sentences)} ({sentence_words} words)")
        
        # Can this sentence fit in current chunk?
        if current_word_count + sentence_words <= max_words:
            current_chunk.append(sentence)
            current_word_count += sentence_words
            print(f"  → Added to current chunk (now {current_word_count}/{max_words} words)")
        else:
            # Save current chunk if it has content
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                print(f"  → Saving chunk #{len(chunks)} with {len(chunk_text.split())} words")
                current_chunk = []
                current_word_count = 0
            
            # Handle oversized single sentence
            if sentence_words > max_words:
                print(f"  ⚠️ Sentence too large ({sentence_words} > {max_words}), splitting at word boundaries")
                words = sentence.split()
                temp_chunk = []
                temp_count = 0
                
                for word in words:
                    if temp_count + 1 <= max_words:
                        temp_chunk.append(word)
                        temp_count += 1
                    else:
                        # Save this word-based chunk
                        if temp_chunk:
                            chunk_text = ' '.join(temp_chunk)
                            chunks.append(chunk_text)
                            print(f"  → Saving word-split chunk #{len(chunks)} with {len(temp_chunk)} words")
                        temp_chunk = [word]
                        temp_count = 1
                
                # Don't forget remaining words
                if temp_chunk:
                    chunk_text = ' '.join(temp_chunk)
                    chunks.append(chunk_text)
                    print(f"  → Saving final word-split chunk #{len(chunks)} with {len(temp_chunk)} words")
            else:
                # Start new chunk with this sentence
                current_chunk = [sentence]
                current_word_count = sentence_words
                print(f"  → Started new chunk with this sentence ({current_word_count} words)")
    
    # CRITICAL: Don't forget the last chunk!
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append(chunk_text)
        print(f"\n→ Saving FINAL chunk #{len(chunks)} with {len(chunk_text.split())} words\n")
    
    # VERIFICATION
    print(f"{'='*60}")
    print(f"CHUNKING COMPLETE")
    print(f"{'='*60}")
    print(f"Total chunks created: {len(chunks)}")
    
    original_word_count = len(text.split())
    chunks_word_count = sum(len(chunk.split()) for chunk in chunks)
    
    print(f"Original word count: {original_word_count}")
    print(f"Chunks total words: {chunks_word_count}")
    
    if original_word_count != chunks_word_count:
        print(f"❌ ERROR: WORD COUNT MISMATCH!")
        print(f"   Lost {original_word_count - chunks_word_count} words!")
        # Log each chunk for debugging
        for i, chunk in enumerate(chunks, 1):
            print(f"   Chunk {i}: {len(chunk.split())} words")
    else:
        print(f"✅ SUCCESS: All {original_word_count} words preserved!")
    
    print(f"{'='*60}\n")
    
    return chunks

# --- TTS Worker ---
@app.function(
    image=worker_image,
    gpu="L4",
    timeout=10800,
    volumes={"/data": vol}
)
def tts_worker(job_id: str, config: dict):
    """Background worker that generates TTS audio."""
    print(f"👷 [Worker] Starting job {job_id}")
    
    sys.path.append("/app")
    import logging
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
    logger = logging.getLogger("tts_worker")
    
    
    import engine
    import torch
    import numpy as np
    import soundfile as sf
    from config import config_manager
    
    job_dir = Path(OUTPUT_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Capture logs to file
    debug_log_path = job_dir / "debug.log"
    file_handler = logging.FileHandler(debug_log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.getLogger("engine").addHandler(file_handler)
    
    status_file = job_dir / "status.json"
    output_file = job_dir / "output.wav"
    
    def update_status(status: str, progress: float, message: str, error: str = None):
        # Read existing status to preserve created_at field
        created_at_time = None
        if status_file.exists():
            try:
                with open(status_file, "r") as f:
                    existing_data = json.load(f)
                    created_at_time = existing_data.get("created_at")
            except:
                pass
        
        # If no created_at exists, set it now
        if created_at_time is None:
            created_at_time = time.time()
        
        data = {
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "message": message,
            "error": error,
            "created_at": created_at_time,  # Persist creation time
            "updated_at": time.time()
        }
        if status == "completed":
            data["completed_at"] = time.time()
        
        with open(status_file, "w") as f:
            json.dump(data, f)
        vol.commit()

    try:
        update_status("processing", 0.0, "Initializing engine...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"👷 [Worker] Using device: {device}")
        
        repo_id = config.get("repo_id", "ResembleAI/chatterbox")
        print(f"👷 [Worker] Configuring model {repo_id}...")
        
        if "model" not in config_manager.config: config_manager.config["model"] = {}
        if "tts_engine" not in config_manager.config: config_manager.config["tts_engine"] = {}
        
        config_manager.config["model"]["repo_id"] = repo_id
        config_manager.config["tts_engine"]["device"] = device
        
        print(f"👷 [Worker] Config: {config_manager.config}")
        print(f"👷 [Worker] Loading model...")
        success = engine.load_model()
        print(f"👷 [Worker] load_model returned: {success}")
        
        if not success:
            raise Exception("Failed to load model")
        
        update_status("processing", 10.0, "Generating audio...")
        
        # Determine voice path
        voice_path = None
        voice_mode = config.get("voice_mode", "predefined")
        text = config.get("text", "")
        
        if voice_mode == "predefined":
            voice_id = config.get("predefined_voice_id", "Olivia.wav")
            # Check mounted voices first
            p = Path("/app/voices") / voice_id
            if p.exists():
                voice_path = p
            else:
                # Check volume
                p = Path(VOICES_DIR) / voice_id
                if p.exists():
                    voice_path = p
                    
        elif voice_mode == "clone":
            ref_file = config.get("reference_file")
            if ref_file:
                p = job_dir / ref_file
                if p.exists():
                    voice_path = p
                else:
                    p = Path(REF_AUDIO_DIR) / ref_file
                    if p.exists():
                        voice_path = p
        
        if not voice_path or not voice_path.exists():
            print(f"⚠️ [Worker] Voice path {voice_path} not found. Using default.")
            voice_path = Path("/app/voices/Olivia.wav")
            if not voice_path.exists():
                raise Exception(f"Default voice file not found at {voice_path}")
            
        print(f"👷 [Worker] Using voice: {voice_path}")

        gen_params = config.get("generation_params", {})
        temperature = float(config.get("temperature", gen_params.get("temperature", 0.7)))
        exaggeration = float(config.get("exaggeration", gen_params.get("exaggeration", 0.5)))
        cfg_weight = float(config.get("cfg_weight", gen_params.get("cfg_weight", 0.5)))
        seed = int(config.get("seed", gen_params.get("seed", 0)))
        speed_factor = float(config.get("speed_factor", gen_params.get("speed_factor", 1.0)))
        
        # CHUNKING FOR LONG TEXT
        # Split text into chunks of 200-250 words to avoid GPU memory issues
        word_count = len(text.split())
        print(f"👷 [Worker] Text word count: {word_count}")
        
        # Use chunk size from request, default to 225 if not provided
        # This allows the UI slider to control the chunk size
        CHUNK_SIZE_WORDS = int(config.get("chunk_size", 225))
        
        if word_count > CHUNK_SIZE_WORDS:
            # Long text - use chunking
            print(f"👷 [Worker] Long text detected. Using chunking ({CHUNK_SIZE_WORDS} words per chunk)")
            chunks = _split_text_by_words(text, CHUNK_SIZE_WORDS)
            print(f"👷 [Worker] Split into {len(chunks)} chunks")
            
            audio_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_num = i + 1
                print(f"👷 [Worker] Processing chunk {chunk_num}/{len(chunks)}: {chunk[:50]}...")
                update_status("processing", 10.0 + (80.0 * chunk_num / len(chunks)), 
                             f"Generating chunk {chunk_num}/{len(chunks)}...")
                
                chunk_audio, chunk_sr = engine.synthesize(
                    chunk,
                    audio_prompt_path=str(voice_path),
                    temperature=temperature,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    seed=seed,
                    speed_factor=speed_factor
                )
                
                if chunk_audio is None:
                    raise Exception(f"Synthesis returned None for chunk {chunk_num}")
                
                # Convert to numpy immediately
                if isinstance(chunk_audio, torch.Tensor):
                    chunk_np = chunk_audio.detach().cpu().numpy()
                else:
                    chunk_np = chunk_audio
                
                chunk_np = chunk_np.squeeze()
                audio_chunks.append(chunk_np)
                
                # Clear GPU memory between chunks (doesn't affect quality, just frees VRAM)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"👷 [Worker] GPU memory cleared after chunk {chunk_num}")
            
            # Concatenate all chunks
            print(f"👷 [Worker] Concatenating {len(audio_chunks)} chunks...")
            audio_np = np.concatenate(audio_chunks, axis=0)
            sr = chunk_sr
            
        else:
            # Short text - single synthesis
            print(f"👷 [Worker] Short text. Using single synthesis.")
            audio_tensor, sr = engine.synthesize(
                text, 
                audio_prompt_path=str(voice_path), 
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                seed=seed,
                speed_factor=speed_factor
            )
            
            if audio_tensor is None:
                raise Exception("Synthesis returned None")
    
            # Convert to numpy and ensure proper shape for soundfile
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.detach().cpu().numpy()
            else:
                audio_np = audio_tensor
            
            audio_np = audio_np.squeeze()
        
        print(f"👷 [Worker] Final audio shape: {audio_np.shape}, dtype: {audio_np.dtype}, sample_rate: {sr}")
        
        # Ensure it's float32 or float64 for soundfile
        if audio_np.dtype.name not in ['float32', 'float64']:
            audio_np = audio_np.astype('float32')
        
        sf.write(str(output_file), audio_np, sr)
        
        # Handle format conversion
        output_format = config.get("output_format", "wav").lower()
        final_output_file = output_file
        
        if output_format == "mp3":
            print(f"👷 [Worker] Converting to MP3...")
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(str(output_file))
                mp3_file = job_dir / "output.mp3"
                audio.export(str(mp3_file), format="mp3", bitrate="192k")
                final_output_file = mp3_file
                print(f"👷 [Worker] Converted to MP3: {final_output_file}")
            except Exception as e:
                print(f"❌ [Worker] MP3 conversion failed: {e}")
                # Fallback to WAV is already saved
        
        print(f"👷 [Worker] Job completed. Saved to {final_output_file}")
        update_status("completed", 100.0, "Audio generation complete.")

        
    except Exception as e:
        print(f"❌ [Worker] Error: {e}")
        import traceback
        traceback.print_exc()
        update_status("failed", 0.0, "Generation failed", str(e))

# --- Web Server ---
web_app = FastAPI(title="Chatterbox Native Queue")

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ui_path = "/app/ui" if os.path.exists("/app/ui") else "ui"
if os.path.exists(ui_path):
    web_app.mount("/ui", StaticFiles(directory=ui_path), name="ui")

@web_app.get("/")
async def index():
    p = Path(ui_path) / "index.html"
    if p.exists():
        return FileResponse(p)
    return HTMLResponse("<h1>Chatterbox TTS</h1>")

@web_app.get("/script.js")
async def script():
    p = Path(ui_path) / "script.js"
    if p.exists():
        return FileResponse(p, media_type="application/javascript")
    return JSONResponse({"error": "Not found"}, status_code=404)

@web_app.get("/styles.css")
async def styles():
    p = Path(ui_path) / "styles.css"
    if p.exists():
        return FileResponse(p, media_type="text/css")
    return JSONResponse({"error": "Not found"}, status_code=404)

@web_app.post("/jobs/submit")
async def submit_job(request: Request):
    """Submit a job to the worker."""
    try:
        data = await request.json()
        text = data.get("text")
        if not text:
            return JSONResponse({"error": "Text is required"}, status_code=400)
            
        job_id = str(uuid.uuid4())
        job_dir = Path(OUTPUT_DIR) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        with open(job_dir / "status.json", "w") as f:
            json.dump({
                "job_id": job_id,
                "status": "queued",
                "progress": 0.0,
                "message": "Job queued",
                "created_at": time.time()
            }, f)
            
        # Pass text in config
        config = data.copy()
        config["text"] = text
        
        tts_worker.spawn(job_id, config)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Job submitted successfully."
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@web_app.get("/jobs/{job_id}/status")
async def get_status(job_id: str):
    """Check job status."""
    status_file = Path(OUTPUT_DIR) / job_id / "status.json"
    vol.reload()
    
    if not status_file.exists():
        if (Path(OUTPUT_DIR) / job_id).exists():
            return {"status": "queued", "message": "Waiting for worker..."}
        return JSONResponse({"error": "Job not found"}, status_code=404)
        
    try:
        with open(status_file, "r") as f:
            return json.load(f)
    except Exception as e:
        return JSONResponse({"error": f"Failed to read status: {e}"}, status_code=500)

@web_app.get("/jobs/{job_id}/result")
async def get_result(job_id: str):
    """Download result file."""
    job_dir = Path(OUTPUT_DIR) / job_id
    vol.reload()
    
    # Check for MP3 first
    mp3_file = job_dir / "output.mp3"
    if mp3_file.exists():
        return FileResponse(mp3_file, media_type="audio/mpeg", filename=f"chatterbox_{job_id}.mp3")
        
    # Check for WAV
    wav_file = job_dir / "output.wav"
    if wav_file.exists():
        return FileResponse(wav_file, media_type="audio/wav", filename=f"chatterbox_{job_id}.wav")
        
    return JSONResponse({"error": "Result not ready"}, status_code=404)

@web_app.get("/jobs/{job_id}/log")
async def get_log(job_id: str):
    """Display debug log with formatted HTML."""
    log_file = Path(OUTPUT_DIR) / job_id / "debug.log"
    vol.reload()
    
    if not log_file.exists():
        html = f"""
        <html>
        <head><title>Log Not Found</title>
        <style>body {{ font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: #fff; }}</style>
        </head>
        <body>
            <h1>Log Not Found</h1>
            <p>No debug log available for job {job_id[:16]}...</p>
            <a href="/manager" style="color: #00aaff;">← Back to Manager</a>
        </body>
        </html>
        """
        return HTMLResponse(html, status_code=404)
    
    with open(log_file, "r") as f:
        log_content = f.read()
    
    html = f"""
    <html>
    <head>
        <title>Job {job_id[:16]}... - Debug Log</title>
        <style>
            body {{ font-family: monospace; margin: 0; background: #0a0a0a; color: #00ff00; }}
            header {{ background: #1a1a1a; padding: 15px 20px; border-bottom: 2px solid #00ff00; position: sticky; top: 0; }}
            h1 {{ margin: 0; font-size: 1.2em; }}
            a {{ color: #00aaff; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            .log-container {{ padding: 20px; max-height: calc(100vh - 80px); overflow-y: auto; }}
            pre {{ background: #000; padding: 20px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; line-height: 1.4; }}
            .download-btn {{ background: #00aa00; color: white; padding: 8px 16px; border-radius: 5px; display: inline-block; margin-left: 20px; }}
            .download-btn:hover {{ background: #00cc00; }}
        </style>
    </head>
    <body>
        <header>
            <h1>Debug Log: {job_id[:16]}...</h1>
            <a href="/manager">← Back to Manager</a>
            <a href="/jobs/{job_id}/log?download=1" class="download-btn" download>📥 Download Log</a>
        </header>
        <div class="log-container">
            <pre>{log_content}</pre>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(html)

@web_app.get("/get_predefined_voices")
async def get_predefined_voices():
    """List available voices."""
    voices = []
    if Path(VOICES_DIR).exists():
        for f in Path(VOICES_DIR).iterdir():
            if f.suffix in ['.wav', '.mp3']:
                voices.append({"filename": f.name, "display_name": f.stem})
    
    app_voices = Path("/app/voices")
    if app_voices.exists():
        for f in app_voices.iterdir():
            if f.suffix in ['.wav', '.mp3']:
                if not any(v['filename'] == f.name for v in voices):
                    voices.append({"filename": f.name, "display_name": f.stem})
    return voices

# --- UI & Production Endpoints ---

@web_app.get("/api/ui/initial-data")
async def get_initial_data():
    """Get initial UI configuration data."""
    # Get reference files
    ref_dir = Path(REF_AUDIO_DIR)
    reference_files = []
    if ref_dir.exists():
        reference_files = [f.name for f in ref_dir.iterdir() if f.suffix in ['.wav', '.mp3']]
    
    return {
        "predefined_voices": await get_predefined_voices(),
        "reference_files": reference_files,
        "presets": [],  # Can be populated later
        "config": {
            "ui": {
                "title": "Chatterbox TTS Server",
                "show_language_select": True
            },
            "generation_defaults": {
                "temperature": 0.7,
                "exaggeration": 0.5,
                "cfg_weight": 0.5,
                "speed_factor": 1.0,
                "seed": 0,
                "language": "en"
            },
            "ui_state": {},
            "tts_engine": {},
            "audio_output": {
                "format": "mp3",
                "sample_rate": 24000
            }
        }
    }

@web_app.post("/save_settings")
async def save_settings(request: Request):
    """Save user settings (stub - settings are per-job in queue mode)."""
    return {"success": True, "message": "Settings saved"}

@web_app.post("/upload_predefined_voice")
async def upload_predefined_voice(files: list[UploadFile] = File(...)):
    """Upload new predefined voice files."""
    uploaded_files = []
    errors = []
    
    for file in files:
        try:
            if not file.filename:
                errors.append({"filename": "unknown", "error": "No filename provided"})
                continue
            
            if not file.filename.lower().endswith(('.wav', '.mp3')):
                errors.append({"filename": file.filename, "error": "Only WAV and MP3 files supported"})
                continue
            
            content = await file.read()
            file_size = len(content)
            
            if file_size > 10 * 1024 * 1024:
                errors.append({"filename": file.filename, "error": "File too large (max 10MB)"})
                continue
            
            voices_dir = Path(VOICES_DIR)
            voices_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = voices_dir / file.filename
            
            with open(file_path, "wb") as f:
                f.write(content)
            
            uploaded_files.append(file.filename)
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})
    
    # Commit volume once after all uploads
    if uploaded_files:
        vol.commit()
    
    # Get updated voices list
    all_voices = await get_predefined_voices()
    
    return {
        "uploaded_files": uploaded_files,
        "errors": errors,
        "all_predefined_voices": all_voices
    }

@web_app.post("/upload_reference")
async def upload_reference(files: list[UploadFile] = File(...)):
    """Upload reference audio files for voice cloning."""
    uploaded_files = []
    errors = []
    
    for file in files:
        try:
            if not file.filename:
                errors.append({"filename": "unknown", "error": "No filename provided"})
                continue
            
            if not file.filename.lower().endswith(('.wav', '.mp3')):
                errors.append({"filename": file.filename, "error": "Only WAV and MP3 files supported"})
                continue
            
            content = await file.read()
            file_size = len(content)
            
            if file_size > 10 * 1024 * 1024:
                errors.append({"filename": file.filename, "error": "File too large (max 10MB)"})
                continue
            
            ref_dir = Path(REF_AUDIO_DIR)
            ref_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = ref_dir / file.filename
            
            with open(file_path, "wb") as f:
                f.write(content)
            
            uploaded_files.append(file.filename)
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})
    
    # Commit volume once after all uploads
    if uploaded_files:
        vol.commit()
    
    # Get updated reference files list
    ref_dir = Path(REF_AUDIO_DIR)
    all_refs = []
    if ref_dir.exists():
        all_refs = [f.name for f in ref_dir.iterdir() if f.suffix in ['.wav', '.mp3']]
    
    return {
        "uploaded_files": uploaded_files,
        "errors": errors,
        "all_reference_files": all_refs
    }

@web_app.get("/vendor/{filename}")
async def vendor_files(filename: str):
    """Serve vendor JavaScript files."""
    vendor_path = Path(ui_path) / "vendor" / filename
    if vendor_path.exists():
        return FileResponse(vendor_path)
    return JSONResponse({"error": "Not found"}, status_code=404)

@web_app.post("/tts")
async def tts_generate(request: Request):
    """Main TTS generation endpoint - synchronous for UI compatibility."""
    import asyncio
    try:
        data = await request.json()
        
        # Validate input
        text = data.get("text", "").strip()
        if not text:
            return JSONResponse({"success": False, "error": "Text is required"}, status_code=400)
        
        if len(text) > 100000:
            return JSONResponse({"success": False, "error": "Text too long (max 100,000 characters)"}, status_code=400)
        
        # Submit to job queue
        job_response = await submit_job(request)
        
        if not isinstance(job_response, dict) or "job_id" not in job_response:
            return JSONResponse({"success": False, "error": "Failed to submit job"}, status_code=500)
        
        job_id = job_response["job_id"]
        
        # Poll for completion (max 5 minutes)
        max_wait = 300  # 5 minutes
        poll_interval = 2  # 2 seconds
        elapsed = 0
        
        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            
            # Reload volume to get latest status
            vol.reload()
            
            # Check job status file
            status_file = Path(OUTPUT_DIR) / job_id / "status.json"
            if status_file.exists():
                try:
                    with open(status_file, "r") as f:
                        status = json.load(f)
                    
                    if status.get("status") == "completed":
                        # Job complete! Return the audio file
                        output_file = Path(OUTPUT_DIR) / job_id / "output.wav"
                        
                        if output_file.exists():
                            with open(output_file, "rb") as f:
                                audio_data = f.read()
                            
                            # Determine content type
                            content_type = "audio/wav"
                            
                            return Response(
                                content=audio_data,
                                media_type=content_type,
                                headers={
                                    "Content-Disposition": f'attachment; filename="chatterbox_{job_id}.wav"'
                                }
                            )
                        else:
                            return JSONResponse({"success": False, "error": "Audio file not found"}, status_code=500)
                    
                    elif status.get("status") == "failed":
                        error_msg = status.get("error", "Unknown error")
                        return JSONResponse({"success": False, "error": f"Generation failed: {error_msg}"}, status_code=500)
                        
                except Exception as e:
                    # Continue polling if can't read status yet
                    pass
        
        # Timeout
        return JSONResponse({"success": False, "error": "Generation timeout (5 minutes exceeded)"}, status_code=504)
        
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@web_app.get("/manager")
async def job_manager():
    """Job manager page with full job listing."""
    import os
    import time as time_module
    
    # Reload volume to get latest jobs
    vol.reload()
    
    # Get all jobs from output directory
    output_path = Path(OUTPUT_DIR)
    jobs = []
    
    if output_path.exists():
        for job_dir in output_path.iterdir():
            if job_dir.is_dir():
                job_id = job_dir.name
                status_file = job_dir / "status.json"
                output_file = job_dir / "output.wav"
                
                # Read status if available
                status = "unknown"
                created_at = "N/A"
                completed_at = "N/A"
                error_msg = None
                
                if status_file.exists():
                    try:
                        with open(status_file, "r") as f:
                            status_data = json.load(f)
                            status = status_data.get("status", "unknown")
                            created_timestamp = status_data.get("created_at", 0)
                            created_at = time_module.strftime('%Y-%m-%d %H:%M:%S', 
                                time_module.localtime(created_timestamp))
                            if status == "completed" and "completed_at" in status_data:
                                completed_at = time_module.strftime('%Y-%m-%d %H:%M:%S',
                                    time_module.localtime(status_data.get("completed_at", 0)))
                            if status == "failed":
                                error_msg = status_data.get("error", "Unknown error")
                    except:
                        # If can't read status, use directory modification time
                        created_timestamp = job_dir.stat().st_mtime
                        created_at = time_module.strftime('%Y-%m-%d %H:%M:%S',
                            time_module.localtime(created_timestamp))
                else:
                    # No status file - use directory creation/modification time
                    created_timestamp = job_dir.stat().st_mtime
                    created_at = time_module.strftime('%Y-%m-%d %H:%M:%S',
                        time_module.localtime(created_timestamp))
                
                # Check if output exists
                has_output = output_file.exists()
                output_size = ""
                if has_output:
                    size_bytes = output_file.stat().st_size
                    output_size = f"{size_bytes / 1024 / 1024:.2f} MB"
                
                jobs.append({
                    "job_id": job_id,
                    "status": status,
                    "created_at": created_at,
                    "created_timestamp": created_timestamp,  # Keep numeric timestamp for sorting
                    "completed_at": completed_at,
                    "has_output": has_output,
                    "output_size": output_size,
                    "error": error_msg
                })
    
    # Sort by created timestamp (newest first) - use numeric timestamp, not string
    jobs.sort(key=lambda x: x.get("created_timestamp", 0), reverse=True)
    
    # Generate jobs HTML
    jobs_html = ""
    if jobs:
        for job in jobs:
            status_color = {
                "completed": "#00ff00",
                "processing": "#ffaa00",
                "queued": "#00aaff",
                "failed": "#ff0000",
                "unknown": "#888888"
            }.get(job["status"], "#888888")
            
            download_btn = ""
            if job["has_output"]:
                download_btn = f'<a href="/jobs/{job["job_id"]}/result" class="btn download">📥 Download ({job["output_size"]})</a>'
            
            error_display = ""
            if job["error"]:
                error_display = f'<div class="error">Error: {job["error"]}</div>'
            
            jobs_html += f'''
            <div class="job-card">
                <div class="job-header">
                    <span class="job-id">Job ID: {job["job_id"][:16]}...</span>
                    <span class="status" style="color: {status_color}">● {job["status"].upper()}</span>
                </div>
                <div class="job-details">
                    <div>Created: {job["created_at"]}</div>
                    {f'<div>Completed: {job["completed_at"]}</div>' if job["completed_at"] != "N/A" else ""}
                </div>
                {error_display}
                <div class="job-actions">
                    <a href="/jobs/{job["job_id"]}/status" class="btn status">📊 Status</a>
                    {download_btn}
                    <a href="/jobs/{job["job_id"]}/log" class="btn log">📄 Logs</a>
                </div>
            </div>
            '''
    else:
        jobs_html = '<div class="no-jobs">No jobs found. Generate some audio to see them here!</div>'
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chatterbox Job Manager</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
                color: #fff;
                min-height: 100vh;
                padding: 20px;
            }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1 {{ 
                font-size: 2.5rem;
                margin-bottom: 10px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            .subtitle {{ color: #aaa; margin-bottom: 30px; font-size: 1.1rem; }}
            .header-actions {{
                display: flex;
                gap: 15px;
                margin-bottom: 30px;
                flex-wrap: wrap;
            }}
            .btn {{
                padding: 10px 20px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 600;
                transition: all 0.3s ease;
                display: inline-block;
                border: 2px solid transparent;
            }}
            .btn:hover {{ transform: translateY(-2px); }}
            .btn.primary {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}
            .btn.primary:hover {{ box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }}
            .btn.secondary {{
                background: rgba(255,255,255,0.1);
                border: 2px solid rgba(255,255,255,0.3);
                color: white;
            }}
            .job-card {{
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 15px;
                transition: all 0.3s ease;
            }}
            .job-card:hover {{
                background: rgba(255,255,255,0.08);
                border-color: rgba(102, 126, 234, 0.5);
                transform: translateX(5px);
            }}
            .job-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }}
            .job-id {{
                font-family: 'Courier New', monospace;
                font-size: 0.9rem;
                color: #aaa;
            }}
            .status {{
                font-weight: 700;
                font-size: 0.9rem;
                padding: 5px 15px;
                background: rgba(0,0,0,0.3);
                border-radius: 20px;
            }}
            .job-details {{
                color: #ccc;
                margin-bottom: 15px;
                font-size: 0.9rem;
            }}
            .job-details div {{ margin-bottom: 5px; }}
            .job-actions {{
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }}
            .job-actions .btn {{
                padding: 8px 16px;
                font-size: 0.9rem;
                background: rgba(255,255,255,0.1);
                color: white;
                border: 1px solid rgba(255,255,255,0.2);
            }}
            .job-actions .btn.download {{
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                border: none;
            }}
            .no-jobs {{
                text-align: center;
                padding: 60px 20px;
                color: #888;
                font-size: 1.2rem;
            }}
            .error {{
                background: rgba(255,0,0,0.2);
                border-left: 3px solid #ff0000;
                padding: 10px;
                margin: 10px 0;
                border-radius: 4px;
                font-size: 0.9rem;
            }}
            @media (max-width: 768px) {{
                h1 {{ font-size: 2rem; }}
                .job-header {{ flex-direction: column; align-items: flex-start; gap: 10px; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Job Manager</h1>
            <div class="subtitle">Track and manage your Chatterbox TTS generations</div>
            
            <div class="header-actions">
                <a href="/" class="btn primary">🎤 Generate New Audio</a>
                <a href="/manager" class="btn secondary" onclick="location.reload()">🔄 Refresh</a>
            </div>
            
            <div class="jobs-container">
                {jobs_html}
            </div>
        </div>
        
        <script>
            // Auto-refresh every 5 seconds if there are active jobs
            const hasActiveJobs = {str(any(j["status"] in ["queued", "processing"] for j in jobs)).lower()};
            if (hasActiveJobs) {{
                setTimeout(() => location.reload(), 5000);
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)

@web_app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "chatterbox-tts",
        "version": "1.0.0",
        "mode": "job-queue"
    }

# --- Entrypoint ---
@app.function(image=web_image, volumes={"/data": vol})
@modal.asgi_app()
def entrypoint():
    return web_app
