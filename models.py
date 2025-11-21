# File: models.py
# Pydantic models for API request and response validation.

from typing import Optional, Literal, Dict, Any, List
from pydantic import BaseModel, Field


class GenerationParams(BaseModel):
    """Common parameters for TTS generation."""

    temperature: Optional[float] = Field(
        None,  # Defaulting to None means server will use config default if not provided
        ge=0.0,
        le=1.5,  # Based on Chatterbox Gradio app for temperature
        description="Controls randomness. Lower is more deterministic. (Range: 0.0-1.5)",
    )
    exaggeration: Optional[float] = Field(
        None,
        ge=0.25,  # Based on Chatterbox Gradio app
        le=2.0,  # Based on Chatterbox Gradio app
        description="Controls expressiveness/exaggeration. (Range: 0.25-2.0)",
    )
    cfg_weight: Optional[float] = Field(
        None,
        ge=0.2,  # Based on Chatterbox Gradio app
        le=1.0,  # Based on Chatterbox Gradio app
        description="Classifier-Free Guidance weight. Influences adherence to prompt/style and pacing. (Range: 0.2-1.0)",
    )
    seed: Optional[int] = Field(
        None,
        ge=0,  # Seed should be non-negative, 0 often implies random.
        description="Seed for generation. 0 may indicate random behavior based on engine.",
    )
    speed_factor: Optional[float] = Field(
        None,
        ge=0.25,
        le=4.0,
        description="Speed factor for the generated audio. 1.0 is normal speed. Applied post-generation.",
    )
    language: Optional[str] = Field(
        None,
        description="Language of the text. (Primarily for UI, actual engine may infer)",
    )


class CustomTTSRequest(BaseModel):
    """Request model for the custom /tts endpoint."""

    text: str = Field(
        ..., 
        min_length=1,
        description="Text to synthesize (maximum 500,000 characters). Will be automatically sanitized and split into chunks."
    )

    voice_mode: Literal["predefined", "clone"] = Field(
        "predefined",
        description="Voice mode: 'predefined' for a built-in voice, 'clone' for voice cloning using a reference audio."
    )
    predefined_voice_id: Optional[str] = Field(
        None,
        description="Filename of the predefined voice to use (e.g., 'default_sample.wav'). Required if voice_mode is 'predefined'."
    )
    reference_audio_filename: Optional[str] = Field(
        None,
        description="Filename of a user-uploaded reference audio for voice cloning (5-120 seconds, max 50MB). Required if voice_mode is 'clone'."
    )

    output_format: Optional[Literal["wav", "opus", "mp3"]] = Field(
        "wav", description="Desired audio output format."
    )

    split_text: Optional[bool] = Field(
        True,
        description="Whether to automatically split long text into chunks for processing."
    )
    chunk_size: Optional[int] = Field(
        150,
        ge=10,
        le=10000,
        description="Target character length for text chunks when splitting (10-10000). Recommended: 150-200 for best voice quality and performance. Lower values increase processing time."
    )

    # Embed generation parameters directly
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0). Lower = more conservative/deterministic, higher = more creative/variable. Default from config if not provided."
    )
    exaggeration: Optional[float] = Field(
        None,
        ge=0.0,
        le=5.0,
        description="Voice expressiveness/exaggeration (0.0-5.0). Higher = more dramatic/expressive. Default from config if not provided."
    )
    cfg_weight: Optional[float] = Field(
        None,
        ge=0.0,
        le=10.0,
        description="Classifier-Free Guidance weight (0.0-10.0). Influences adherence to prompt/style. Default from config if not provided."
    )
    seed: Optional[int] = Field(
        None,
        ge=0,
        le=2147483647,
        description="Random seed for reproducibility (0-2147483647). None = random generation, integer = fixed/reproducible output."
    )
    speed_factor: Optional[float] = Field(
        None,
        ge=0.5,
        le=2.0,
        description="Audio playback speed multiplier (0.5-2.0). 1.0 = normal speed, <1.0 = slower, >1.0 = faster. Applied post-generation."
    )
    language: Optional[str] = Field(
        None, description="Language of the text (primarily for UI, engine may auto-detect)."
    )
    
    allow_partial_success: Optional[bool] = Field(
        True,
        description="If True, skips chunks that fail synthesis and continues. Failed chunks logged for manual regeneration. If False, process exits on first failure."
    )



class ErrorResponse(BaseModel):
    """Standard error response model for API errors."""

    detail: str = Field(..., description="A human-readable explanation of the error.")


class UpdateStatusResponse(BaseModel):
    """Response model for status updates, e.g., after saving settings."""

    success: bool = Field(..., description="Indicates if the operation was successful.")
    message: str = Field(
        ..., description="A message describing the result of the operation."
    )
    restart_required: Optional[bool] = Field(
        False,
        description="Indicates if a server restart is recommended or required for changes to take full effect.",
    )


class UIInitialDataResponse(BaseModel):
    """
    Response model for UI bootstrap data.
    """
    reference_files: List[str]
    predefined_voices: List[Dict[str, Any]]
    ui_state: Dict[str, Any]
    presets: List[Dict[str, Any]]


class FileListResponse(BaseModel):
    """
    Response model for file listing.
    """
    files: List[str] = []


# --------------------------------------------------------------------------------------
# Job Queue Models (for async TTS processing)
# --------------------------------------------------------------------------------------

class JobSubmissionResponse(BaseModel):
    """Response when a TTS job is submitted for async processing."""
    job_id: str = Field(..., description="Unique job ID for tracking synthesis progress")
    status: str = Field(..., description="Initial job status (usually 'queued')")
    message: str = Field(..., description="Human-readable confirmation message")
    estimated_chunks: Optional[int] = Field(None, description="Estimated number of chunks to process")


class JobStatusResponse(BaseModel):
    """Response when checking job status."""
    job_id: str = Field(..., description="Job ID")
    status: str = Field(..., description="Current status: queued, processing, completed, failed, cancelled")
    progress_percent: float = Field(..., description="Progress percentage (0-100)")
    current_step: str = Field(..., description="Human-readable current step description")
    
    # Optional details
    processed_chunks: Optional[int] = Field(None, description="Number of chunks processed")
    total_chunks: Optional[int] = Field(None, description="Total number of chunks")
    
    # Timestamps
    created_at: str = Field(..., description="Job creation timestamp (ISO format)")
    started_at: Optional[str] = Field(None, description="Job start timestamp (ISO format)")
    completed_at: Optional[str] = Field(None, description="Job completion timestamp (ISO format)")
    
    # Result/error info
    result_available: bool = Field(False, description="True if result is ready to download")
    error_message: Optional[str] = Field(None, description="Error message if job failed")
    failed_chunks_count: Optional[int] = Field(None, description="Number of chunks that failed (if partial success)")


class JobResultResponse(BaseModel):
    """Response when retrieving completed job result."""
    job_id: str = Field(..., description="Job ID")
    status: str = Field(..., description="Job status")
    result_file: Optional[str] = Field(None, description="Path/filename of generated audio file")
    error_message: Optional[str] = Field(None, description="Error message if job failed")
    failed_chunks: Optional[List[Dict[str, Any]]] = Field(None, description="Info about failed chunks if partial success")
