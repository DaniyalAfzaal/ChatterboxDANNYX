# File: server.py
# Main FastAPI application for the TTS Server.
# Handles API requests for text-to-speech generation, UI serving,
# configuration management, and file uploads.

import os
import io
import logging
import logging.handlers  # For RotatingFileHandler
import shutil
import time
import uuid
import threading
import requests
import yaml  # For loading presets
import numpy as np
import librosa  # For potential direct use if needed, though utils.py handles most
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    BackgroundTasks,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
    FileResponse,
    PlainTextResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import engine
from config import (
    config_manager,
    get_log_file_path,
    get_output_path,
    get_reference_audio_path,
    get_predefined_voices_path,
    get_audio_output_format,
    get_audio_sample_rate,
    get_host,
    get_port,
    get_ui_title,
    get_model_repo_id,
    get_model_cache_path,
    # get_device_setting,  # 🔥 removed
    get_gen_default_speed_factor,
    get_default_split_text_setting,
    get_default_chunk_size,
    get_gen_default_temperature,
    get_gen_default_exaggeration,
    get_gen_default_cfg_weight,
    get_gen_default_seed,
)
from models import (
    CustomTTSRequest,
    UIInitialDataResponse,
    FileListResponse,
    ErrorResponse,
    UpdateStatusResponse,
)
import utils


class OpenAISpeechRequest(BaseModel):
    model: str
    input_: str = Field(..., alias="input")
    voice: str
    response_format: Literal["wav", "opus", "mp3"] = "wav"
    speed: float = 1.0
    seed: Optional[int] = None


# --------------------------------------------------------------------------------------
# Logging Configuration
# --------------------------------------------------------------------------------------

log_file_path = get_log_file_path()
log_dir = os.path.dirname(log_file_path)
os.makedirs(log_dir, exist_ok=True)

file_handler = logging.handlers.RotatingFileHandler(
    log_file_path,
    maxBytes=config_manager.get_int("logging.max_bytes", 5 * 1024 * 1024),
    backupCount=config_manager.get_int("logging.backup_count", 3),
    encoding="utf-8",
)

formatter = logging.Formatter(
    fmt=config_manager.get_string(
        "logging.format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    ),
    datefmt=config_manager.get_string("logging.datefmt", "%Y-%m-%d %H:%M:%S"),
)

file_handler.setFormatter(formatter)

logging.basicConfig(
    level=config_manager.get_string("logging.level", "INFO"),
    handlers=[file_handler, logging.StreamHandler()],
)

logger = logging.getLogger("tts_server")

# --------------------------------------------------------------------------------------
# FastAPI Initialization with Lifespan
# --------------------------------------------------------------------------------------

templates = Jinja2Templates(directory="ui")

from threading import Event

startup_complete_event = Event()
model_loaded_event = Event()


def _delayed_browser_open(host: str, port: int):
    try:
        import webbrowser

        for _ in range(10):
            if engine.MODEL_LOADED:
                break
            time.sleep(1.5)
        display_host = "localhost" if host == "0.0.0.0" else host
        browser_url = f"http://{display_host}:{port}/"
        logger.info(f"Attempting to open web browser to: {browser_url}")
        webbrowser.open(browser_url)
    except Exception as e:
        logger.error(f"Failed to open browser automatically: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TTS Server: Initializing...")

    output_path = get_output_path(ensure_absolute=True)
    reference_audio_path = get_reference_audio_path(ensure_absolute=True)
    predefined_voices_path = get_predefined_voices_path(ensure_absolute=True)

    for path in [output_path, reference_audio_path, predefined_voices_path]:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {path}")

    ui_dir = Path("ui")
    if not ui_dir.exists():
        logger.warning("UI directory not found. Creating empty 'ui' directory.")
        ui_dir.mkdir(parents=True, exist_ok=True)

    model_cache_path = get_model_cache_path(ensure_absolute=True)
    model_cache_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Model cache directory: {model_cache_path}")

    try:
        if not engine.load_model():
            logger.critical("Failed to load TTS model during startup.")
        else:
            logger.info("TTS Model loaded successfully via engine.")
            model_loaded_event.set()
    except Exception as e:
        logger.exception(f"Exception during model loading: {e}")

    try:
        server_host = get_host()
        server_port = get_port()
        import threading as _threading

        browser_thread = _threading.Thread(
            target=lambda: _delayed_browser_open(server_host, server_port),
            daemon=True,
        )
        browser_thread.start()
    except Exception as e:
        logger.error(f"Error starting browser thread: {e}")

    startup_complete_event.set()
    logger.info("Startup sequence completed.")
    yield
    logger.info("Application shutdown sequence initiated.")
    logger.info("Application shutdown sequence completed.")


app = FastAPI(
    title=get_ui_title(),
    description="Text-to-Speech server with advanced UI and API capabilities.",
    version="2.0.2",
    lifespan=lifespan,
)

# --------------------------------------------------------------------------------------
# CORS and Static Files
# --------------------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory="ui"), name="ui")

outputs_static_path = get_output_path(ensure_absolute=True)
if outputs_static_path.exists():
    app.mount("/outputs", StaticFiles(directory=str(outputs_static_path)), name="outputs")
else:
    logger.warning(
        f"Outputs directory {outputs_static_path} does not exist; skipping static mount."
    )

vendor_dir = Path("ui") / "vendor"
if vendor_dir.exists():
    app.mount("/vendor", StaticFiles(directory=str(vendor_dir)), name="vendor")
else:
    logger.info("No vendor directory found under ui/. Skipping vendor static mount.")

# --------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------


def load_presets() -> Dict[str, Any]:
    presets_path = Path("ui") / "presets.yaml"
    if not presets_path.is_file():
        logger.info("No presets.yaml found in ui/. Using empty presets.")
        return {}
    try:
        with open(presets_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load presets.yaml: {e}")
        return {}


def list_reference_files() -> List[str]:
    ref_dir = get_reference_audio_path(ensure_absolute=True)
    if not ref_dir.exists():
        return []
    return utils.get_valid_reference_files(ref_dir)


def list_predefined_voices() -> List[str]:
    voices_dir = get_predefined_voices_path(ensure_absolute=True)
    if not voices_dir.exists():
        return []
    return utils.get_valid_reference_files(voices_dir)


def get_ui_state_from_config() -> Dict[str, Any]:
    ui_state = config_manager.get_dict("ui_state", default={})
    if "last_selected_mode" not in ui_state:
        ui_state["last_selected_mode"] = "clone"
    if "last_selected_predefined_voice" not in ui_state:
        ui_state["last_selected_predefined_voice"] = ""
    if "last_selected_reference_file" not in ui_state:
        ui_state["last_selected_reference_file"] = ""
    if "split_text" not in ui_state:
        ui_state["split_text"] = get_default_split_text_setting()
    if "chunk_size" not in ui_state:
        ui_state["chunk_size"] = get_default_chunk_size()
    return ui_state


# --------------------------------------------------------------------------------------
# Routes: UI and Static
# --------------------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    index_path = Path("ui") / "index.html"
    if not index_path.is_file():
        return HTMLResponse(
            content="<h1>UI not found</h1><p>index.html missing in ui/ directory.</p>",
            status_code=404,
        )
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/styles.css", response_class=PlainTextResponse)
async def styles():
    css_path = Path("ui") / "styles.css"
    if not css_path.is_file():
        return PlainTextResponse(content="", status_code=404)
    with open(css_path, "r", encoding="utf-8") as f:
        css_content = f.read()
    return PlainTextResponse(content=css_content, status_code=200)


@app.get("/script.js", response_class=PlainTextResponse)
async def script():
    js_path = Path("ui") / "script.js"
    if not js_path.is_file():
        return PlainTextResponse(content="// script.js not found", status_code=404)
    with open(js_path, "r", encoding="utf-8") as f:
        js_content = f.read()
    return PlainTextResponse(content=js_content, status_code=200)


# --------------------------------------------------------------------------------------
# Routes: API - Initial UI Data
# --------------------------------------------------------------------------------------


@app.get("/api/ui/initial-data", response_model=UIInitialDataResponse)
async def get_ui_initial_data():
    try:
        reference_files = list_reference_files()
        predefined_voice_files = list_predefined_voices()
        ui_state = get_ui_state_from_config()
        presets = load_presets()

        return UIInitialDataResponse(
            reference_files=reference_files,
            predefined_voices=predefined_voice_files,
            ui_state=ui_state,
            presets=presets,
        )
    except Exception as e:
        logger.exception(f"Error building UI initial data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to load UI initial data.",
        )


# --------------------------------------------------------------------------------------
# Routes: Configuration Management
# --------------------------------------------------------------------------------------


@app.post("/save_settings", response_model=UpdateStatusResponse)
async def save_settings(new_settings: Dict[str, Any]):
    try:
        logger.info(f"Received settings update: {new_settings}")
        config_manager.update(new_settings)
        config_manager.save()
        restart_flag = True
        return UpdateStatusResponse(
            success=True,
            message="Settings saved successfully. A server restart may be required.",
            restart_required=restart_flag,
        )
    except Exception as e:
        logger.exception(f"Failed to save settings: {e}")
        return UpdateStatusResponse(
            success=False,
            message=f"Failed to save settings: {e}",
            restart_required=False,
        )


@app.post("/reset_settings", response_model=UpdateStatusResponse)
async def reset_settings():
    try:
        config_manager.reset_and_save()
        return UpdateStatusResponse(
            success=True,
            message="Settings reset to default successfully.",
            restart_required=True,
        )
    except Exception as e:
        logger.exception(f"Failed to reset settings: {e}")
        return UpdateStatusResponse(
            success=False,
            message=f"Failed to reset settings: {e}",
            restart_required=False,
        )


@app.post("/restart_server", response_model=UpdateStatusResponse)
async def restart_server():
    logger.warning(
        "Received request to restart server. This is a placeholder in this implementation."
    )
    return UpdateStatusResponse(
        success=True,
        message="Server restart requested. Please restart the process externally if needed.",
        restart_required=True,
    )


# --------------------------------------------------------------------------------------
# Routes: File Management
# --------------------------------------------------------------------------------------


@app.get("/get_reference_files", response_model=FileListResponse)
async def get_reference_files():
    try:
        files = list_reference_files()
        return FileListResponse(files=files)
    except Exception as e:
        logger.exception(f"Failed to list reference files: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list reference files.",
        )


@app.get("/get_predefined_voices", response_model=FileListResponse)
async def get_predefined_voices():
    try:
        files = list_predefined_voices()
        return FileListResponse(files=files)
    except Exception as e:
        logger.exception(f"Failed to list predefined voices: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list predefined voices.",
        )


@app.post("/upload_reference")
async def upload_reference(files: List[UploadFile] = File(...)):
    ref_dir = get_reference_audio_path(ensure_absolute=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    errors = []

    for file in files:
        try:
            filename = utils.sanitize_filename(file.filename)
            dest_path = ref_dir / filename
            with open(dest_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(filename)
            logger.info(f"Uploaded reference audio: {filename}")
        except Exception as e:
            logger.exception(f"Failed to save uploaded reference file {file.filename}: {e}")
            errors.append(f"Failed to save {file.filename}: {e}")
        finally:
            await file.close()

    return JSONResponse(
        content={
            "uploaded_files": saved_files,
            "errors": errors,
        },
        status_code=200 if saved_files else 400,
    )


@app.post("/upload_predefined_voice")
async def upload_predefined_voice(files: List[UploadFile] = File(...)):
    voices_dir = get_predefined_voices_path(ensure_absolute=True)
    voices_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    errors = []

    for file in files:
        try:
            filename = utils.sanitize_filename(file.filename)
            dest_path = voices_dir / filename
            with open(dest_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(filename)
            logger.info(f"Uploaded predefined voice: {filename}")
        except Exception as e:
            logger.exception(f"Failed to save uploaded predefined voice {file.filename}: {e}")
            errors.append(f"Failed to save {file.filename}: {e}")
        finally:
            await file.close()

    return JSONResponse(
        content={
            "uploaded_files": saved_files,
            "errors": errors,
        },
        status_code=200 if saved_files else 400,
    )


# --------------------------------------------------------------------------------------
# Routes: Core TTS Generation
# --------------------------------------------------------------------------------------


@app.post("/tts")
async def custom_tts_endpoint(
    request: CustomTTSRequest, background_tasks: BackgroundTasks
):
    """
    Generates speech audio from text using specified parameters.
    Handles various voice modes (predefined, clone) and audio processing options.
    Returns audio as a stream (WAV or Opus).
    """
    perf_monitor = utils.PerformanceMonitor(
        enabled=config_manager.get_bool("server.enable_performance_monitor", False)
    )
    perf_monitor.record("TTS request received")

    if not engine.MODEL_LOADED:
        logger.error("TTS request failed: Model not loaded.")
        raise HTTPException(
            status_code=500,
            detail="TTS engine model is not currently loaded or available.",
        )

    logger.info(
        f"Received /tts request: mode='{request.voice_mode}', format='{request.output_format}'"
    )
    logger.debug(
        f"TTS params: seed={request.seed}, split={request.split_text}, chunk_size={request.chunk_size}"
    )
    logger.debug(f"Input text (first 100 chars): '{request.text[:100]}...'")

    audio_prompt_path_for_engine: Optional[Path] = None
    if request.voice_mode == "predefined":
        if not request.predefined_voice_id:
            raise HTTPException(
                status_code=400,
                detail="Predefined voice ID must be provided in 'predefined' mode.",
            )
        voices_dir = get_predefined_voices_path(ensure_absolute=True)
        potential_path = voices_dir / request.predefined_voice_id
        if not potential_path.is_file():
            logger.error(f"Predefined voice file not found: {potential_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Predefined voice file '{request.predefined_voice_id}' not found.",
            )
        max_dur = config_manager.get_int("audio_output.max_reference_duration_sec", 30)
        is_valid, msg = utils.validate_reference_audio(potential_path, max_dur)
        if not is_valid:
            raise HTTPException(
                status_code=400, detail=f"Invalid predefined voice file: {msg}"
            )
        audio_prompt_path_for_engine = potential_path
        logger.info(f"Using predefined voice: {request.predefined_voice_id}")

    elif request.voice_mode == "clone":
        if not request.reference_audio_filename:
            raise HTTPException(
                status_code=400,
                detail="Reference audio filename must be provided in 'clone' mode.",
            )
        ref_dir = get_reference_audio_path(ensure_absolute=True)
        potential_path = ref_dir / request.reference_audio_filename
        if not potential_path.is_file():
            logger.error(
                f"Reference audio file for cloning not found: {potential_path}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"Reference audio file '{request.reference_audio_filename}' not found.",
            )
        max_dur = config_manager.get_int("audio_output.max_reference_duration_sec", 30)
        is_valid, msg = utils.validate_reference_audio(potential_path, max_dur)
        if not is_valid:
            raise HTTPException(
                status_code=400, detail=f"Invalid reference audio: {msg}"
            )
        audio_prompt_path_for_engine = potential_path
        logger.info(
            f"Using reference audio for cloning: {request.reference_audio_filename}"
        )

    # Text splitting
    text_chunks: List[str] = []
    if request.split_text and len(request.text) > (
        request.chunk_size or get_default_chunk_size()
    ):
        chunk_size = request.chunk_size or get_default_chunk_size()
        logger.info(
            f"Splitting text into chunks of approx {chunk_size} chars, "
            f"using sentence-aware logic."
        )
        text_chunks = utils.chunk_text_by_sentences(request.text, chunk_size)
    else:
        text_chunks = [request.text]
        logger.info(
            "Processing text as a single chunk (splitting not enabled or text too short)."
        )

    if not text_chunks:
        raise HTTPException(
            status_code=400, detail="Text processing resulted in no usable chunks."
        )

    all_audio_segments_np: List[np.ndarray] = []
    final_output_sample_rate = get_audio_sample_rate()
    engine_output_sample_rate: Optional[int] = None

    for i, chunk in enumerate(text_chunks):
        logger.info(f"Synthesizing chunk {i+1}/{len(text_chunks)}...")
        try:
            chunk_audio_tensor, chunk_sr_from_engine = engine.synthesize(
                text=chunk.strip(),
                audio_prompt_path=(
                    str(audio_prompt_path_for_engine)
                    if audio_prompt_path_for_engine
                    else None
                ),
                temperature=(
                    request.temperature
                    if request.temperature is not None
                    else get_gen_default_temperature()
                ),
                exaggeration=(
                    request.exaggeration
                    if request.exaggeration is not None
                    else get_gen_default_exaggeration()
                ),
                cfg_weight=(
                    request.cfg_weight
                    if request.cfg_weight is not None
                    else get_gen_default_cfg_weight()
                ),
                seed=(
                    request.seed
                    if request.seed is not None
                    else get_gen_default_seed()
                ),
            )

            if chunk_audio_tensor is None or chunk_sr_from_engine is None:
                raise RuntimeError("Engine returned no audio for this chunk.")

            if engine_output_sample_rate is None:
                engine_output_sample_rate = chunk_sr_from_engine
                logger.info(
                    f"Engine sample rate set from first chunk: {engine_output_sample_rate}Hz"
                )
            elif engine_output_sample_rate != chunk_sr_from_engine:
                logger.warning(
                    f"Inconsistent sample rate from engine: chunk {i+1} ({chunk_sr_from_engine}Hz) "
                    f"differs from previous ({engine_output_sample_rate}Hz). Using first chunk's SR."
                )

            current_processed_audio_tensor = chunk_audio_tensor

            speed_factor_to_use = (
                request.speed_factor
                if request.speed_factor is not None
                else get_gen_default_speed_factor()
            )
            if speed_factor_to_use != 1.0:
                current_processed_audio_tensor, _ = utils.apply_speed_factor(
                    current_processed_audio_tensor,
                    chunk_sr_from_engine,
                    speed_factor_to_use,
                )
                perf_monitor.record(f"Speed factor applied to chunk {i+1}")

            processed_audio_np = current_processed_audio_tensor.cpu().numpy().squeeze()
            all_audio_segments_np.append(processed_audio_np)

        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.exception(f"Error while processing chunk {i+1}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during TTS generation for chunk {i+1}: {e}",
            )

    if not all_audio_segments_np:
        logger.error("No audio chunks were generated.")
        raise HTTPException(
            status_code=500,
            detail="No audio was generated from the provided text.",
        )

    try:
        final_audio_np = np.concatenate(all_audio_segments_np)
        perf_monitor.record("Audio concatenation completed")
    except Exception as e_concat:
        logger.exception(f"Error concatenating audio segments: {e_concat}")
        for idx, seg in enumerate(all_audio_segments_np):
            logger.error(f"Segment {idx} shape: {seg.shape}, dtype: {seg.dtype}")
        raise HTTPException(
            status_code=500, detail=f"Audio concatenation error: {e_concat}"
        )

    output_format_str = (
        request.output_format if request.output_format else get_audio_output_format()
    )

    encoded_audio_bytes = utils.encode_audio(
        audio_array=final_audio_np,
        sample_rate=engine_output_sample_rate,
        output_format=output_format_str,
        target_sample_rate=final_output_sample_rate,
    )
    perf_monitor.record(
        f"Final audio encoded to {output_format_str} (target SR: {final_output_sample_rate}Hz "
        f"from engine SR: {engine_output_sample_rate}Hz)"
    )

    if encoded_audio_bytes is None or len(encoded_audio_bytes) < 100:
        logger.error(
            f"Failed to encode final audio to format: {output_format_str} "
            f"or output is too small ({len(encoded_audio_bytes or b'')} bytes)."
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to encode audio to {output_format_str} or generated invalid audio.",
        )

    media_type = f"audio/{output_format_str}"
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    suggested_filename_base = f"tts_output_{timestamp_str}"
    download_filename = utils.sanitize_filename(
        f"{suggested_filename_base}.{output_format_str}"
    )
    headers = {"Content-Disposition": f'attachment; filename="{download_filename}"'}

    # Save generated audio to disk in the configured output folder
    try:
        output_dir = get_output_path(ensure_absolute=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir / download_filename

        with open(output_file_path, "wb") as f:
            f.write(encoded_audio_bytes)

        logger.info(f"Saved generated audio file to: {output_file_path}")
    except Exception as e_save:
        logger.error(f"Failed to save generated audio to disk: {e_save}")

    logger.info(
        f"Successfully generated audio: {download_filename}, {len(encoded_audio_bytes)} bytes, type {media_type}."
    )
    logger.debug(perf_monitor.report())

    return StreamingResponse(
        io.BytesIO(encoded_audio_bytes), media_type=media_type, headers=headers
    )


# --------------------------------------------------------------------------------------
# Background Job Endpoint: /tts_job
# --------------------------------------------------------------------------------------


def _run_tts_job_worker(payload: dict):
    """
    Internal worker that calls the existing /tts endpoint from within the server
    so that long-form generation can proceed even after the original client
    disconnects. The /tts endpoint is responsible for saving the final audio
    file into the configured outputs directory.
    """
    try:
        port = get_port()
        base_url = f"http://127.0.0.1:{port}"
        resp = requests.post(f"{base_url}/tts", json=payload, stream=True, timeout=None)

        for _ in resp.iter_content(chunk_size=1024 * 1024):
            pass

        logger.info(
            f"/tts_job worker finished with status {resp.status_code} "
            f"and content-length={resp.headers.get('Content-Length')}"
        )
    except Exception as e:
        logger.error(f"/tts_job worker failed: {e}", exc_info=True)


@app.post("/tts_job")
async def tts_job_endpoint(request: CustomTTSRequest):
    """
    Starts a background TTS job using the same pipeline as /tts, but decoupled
    from the client connection. The request returns quickly with a job_id while
    the actual synthesis continues in a background worker.

    The generated audio file will be saved into the outputs directory by the
    /tts endpoint, and can be accessed later (e.g., via your Modal file manager).
    """
    job_payload = request.dict()
    job_id = str(uuid.uuid4())

    worker_thread = threading.Thread(
        target=_run_tts_job_worker,
        args=(job_payload,),
        daemon=True,
    )
    worker_thread.start()

    logger.info(f"Started background TTS job with id={job_id}")
    return {"job_id": job_id, "status": "started"}


# --------------------------------------------------------------------------------------
# OpenAI-Compatible Endpoint
# --------------------------------------------------------------------------------------


@app.post("/v1/audio/speech", tags=["OpenAI Compatible"])
async def openai_speech_endpoint(request: OpenAISpeechRequest):
    if not engine.MODEL_LOADED:
        logger.error("OpenAI TTS request failed: Model not loaded.")
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded. Please check server logs.",
        )

    if not request.input_ or not request.input_.strip():
        raise HTTPException(
            status_code=400,
            detail="Text input cannot be empty.",
        )

    voice_name = request.voice
    audio_prompt_path_for_engine: Optional[Path] = None

    predefined_dir = get_predefined_voices_path(ensure_absolute=True)
    ref_dir = get_reference_audio_path(ensure_absolute=True)

    predefined_candidate = predefined_dir / voice_name
    reference_candidate = ref_dir / voice_name

    if predefined_candidate.is_file():
        audio_prompt_path_for_engine = predefined_candidate
        logger.info(f"OpenAI TTS using predefined voice file: {predefined_candidate}")
    elif reference_candidate.is_file():
        audio_prompt_path_for_engine = reference_candidate
        logger.info(f"OpenAI TTS using reference audio file: {reference_candidate}")
    else:
        logger.warning(
            f"OpenAI TTS: voice file '{voice_name}' not found in predefined or reference directories."
        )

    try:
        audio_tensor, sr_from_engine = engine.synthesize(
            text=request.input_,
            audio_prompt_path=(
                str(audio_prompt_path_for_engine)
                if audio_prompt_path_for_engine
                else None
            ),
            temperature=0.7,
            exaggeration=0.0,
            cfg_weight=3.0,
            seed=request.seed or 0,
        )
    except Exception as e:
        logger.exception(f"OpenAI-compatible TTS failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI-compatible TTS failed: {e}",
        )

    if audio_tensor is None or sr_from_engine is None:
        raise HTTPException(
            status_code=500,
            detail="Engine returned no audio.",
        )

    if request.speed != 1.0:
        audio_tensor, _ = utils.apply_speed_factor(
            audio_tensor, sr_from_engine, request.speed
        )

    audio_np = audio_tensor.cpu().numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.squeeze()

    encoded_audio = utils.encode_audio(
        audio_array=audio_np,
        sample_rate=sr_from_engine,
        output_format=request.response_format,
        target_sample_rate=get_audio_sample_rate(),
    )

    if encoded_audio is None or len(encoded_audio) < 100:
        raise HTTPException(
            status_code=500,
            detail="Failed to encode audio or generated invalid audio.",
        )

    media_type = f"audio/{request.response_format}"
    return StreamingResponse(
        io.BytesIO(encoded_audio),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="speech.{request.response_format}"'
        },
    )


# --------------------------------------------------------------------------------------
# Main Entrypoint (for local runs)
# --------------------------------------------------------------------------------------


if __name__ == "__main__":
    server_host = get_host()
    server_port = get_port()
    logger.info(f"Starting TTS server on {server_host}:{server_port}")
    import uvicorn

    uvicorn.run(
        "server:app",
        host=server_host,
        port=server_port,
        log_level="info",
        workers=1,
        reload=False,
    )
