# File: validation.py
# Centralized input validation for TTS server
# Validates all user inputs to prevent errors and security issues

from typing import Optional
from pathlib import Path
import numpy as np
import librosa
from fastapi import HTTPException


class ValidationError(HTTPException):
    """Custom validation error with 400 status code"""
    def __init__(self, message: str):
        super().__init__(status_code=400, detail=message)


def validate_chunk_size(size: Optional[int]) -> int:
    """
    Validate chunk size parameter.
    
    Args:
        size: Chunk size in characters (None uses default)
        
    Returns:
        Validated chunk size
        
    Raises:
        ValidationError: If chunk size is invalid
    """
    if size is None:
        return 150  # Default
    
    if not isinstance(size, int):
        raise ValidationError(f"chunk_size must be an integer, got {type(size).__name__}")
    
    if size <= 0:
        raise ValidationError(f"chunk_size must be positive, got {size}")
    
    if size < 10:
        raise ValidationError(f"chunk_size must be at least 10 (got {size}). Very small chunks are inefficient.")
    
    if size > 10000:
        raise ValidationError(f"chunk_size must not exceed 10,000 (got {size}). Use smaller chunks for stability.")
    
    return size


def validate_speed_factor(factor: Optional[float]) -> float:
    """
    Validate speed factor parameter.
    
    Args:
        factor: Speed multiplier (None uses default)
        
    Returns:
        Validated speed factor
        
    Raises:
        ValidationError: If speed factor is invalid
    """
    if factor is None:
        return 1.0  # Default
    
    if not isinstance(factor, (int, float)):
        raise ValidationError(f"speed_factor must be a number, got {type(factor).__name__}")
    
    factor = float(factor)
    
    if factor <= 0:
        raise ValidationError(f"speed_factor must be positive, got {factor}")
    
    if factor < 0.5:
        raise ValidationError(f"speed_factor too slow (minimum 0.5), got {factor}")
    
    if factor > 2.0:
        raise ValidationError(f"speed_factor too fast (maximum 2.0), got {factor}")
    
    return factor


def validate_temperature(temp: Optional[float]) -> float:
    """
    Validate temperature parameter.
    
    Args:
        temp: Sampling temperature (None uses default)
        
    Returns:
        Validated temperature
        
    Raises:
        ValidationError: If temperature is invalid
    """
    if temp is None:
        return 1.0  # Default
    
    if not isinstance(temp, (int, float)):
        raise ValidationError(f"temperature must be a number, got {type(temp).__name__}")
    
    temp = float(temp)
    
    if temp < 0:
        raise ValidationError(f"temperature must be non-negative, got {temp}")
    
    if temp > 2.0:
        raise ValidationError(f"temperature too high (maximum 2.0), got {temp}. High values may cause instability.")
    
    return temp


def validate_cfg_weight(weight: Optional[float]) -> float:
    """
    Validate CFG weight parameter.
    
    Args:
        weight: Classifier-free guidance weight (None uses default)
        
    Returns:
        Validated CFG weight
        
    Raises:
        ValidationError: If CFG weight is invalid
    """
    if weight is None:
        return 0.5  # Changed from 3.0 to match engine.py default
    
    if not isinstance(weight, (int, float)):
        raise ValidationError(f"cfg_weight must be a number, got {type(weight).__name__}")
    
    weight = float(weight)
    
    if weight < 0:
        raise ValidationError(f"cfg_weight must be non-negative, got {weight}")
    
    if weight > 10.0:
        raise ValidationError(f"cfg_weight too high (maximum 10.0), got {weight}")
    
    return weight


def validate_exaggeration(exag: Optional[float]) -> float:
    """
    Validate exaggeration parameter.
    
    Args:
        exag: Voice exaggeration factor (None uses default)
        
    Returns:
        Validated exaggeration
        
    Raises:
        ValidationError: If exaggeration is invalid
    """
    if exag is None:
        return 1.0  # Default
    
    if not isinstance(exag, (int, float)):
        raise ValidationError(f"exaggeration must be a number, got {type(exag).__name__}")
    
    exag = float(exag)
    
    if exag < 0:
        raise ValidationError(f"exaggeration must be non-negative, got {exag}")
    
    if exag > 5.0:
        raise ValidationError(f"exaggeration too high (maximum 5.0), got {exag}")
    
    return exag


def validate_seed(seed: Optional[int]) -> Optional[int]:
    """
    Validate random seed parameter.
    
    Args:
        seed: Random seed (None means random)
        
    Returns:
        Validated seed or None
        
    Raises:
        ValidationError: If seed is invalid
    """
    if seed is None:
        return None  # Random
    
    if not isinstance(seed, int):
        raise ValidationError(f"seed must be an integer or None, got {type(seed).__name__}")
    
    if seed < 0:
        raise ValidationError(f"seed must be non-negative, got {seed}")
    
    if seed > 2**31 - 1:
        raise ValidationError(f"seed too large (maximum {2**31-1}), got {seed}")
    
    return seed


def validate_text_input(text: str) -> str:
    """
    Validate and clean text input.
    
    Args:
        text: Input text to synthesize
        
    Returns:
        Cleaned and validated text
        
    Raises:
        ValidationError: If text is invalid
    """
    if not isinstance(text, str):
        raise ValidationError(f"text must be a string, got {type(text).__name__}")
    
    # UTF-8 validation
    try:
        text.encode('utf-8').decode('utf-8')
    except UnicodeError as e:
        raise ValidationError(f"Invalid UTF-8 encoding: {e}")
    
    # Length check before cleaning
    if len(text) > 500000:
        raise ValidationError(
            f"Text too long: {len(text)} characters (maximum 500,000). "
            f"Consider breaking into multiple requests."
        )
    
    # Strip emoji (optional - can be configurable)
    try:
        import emoji
        original_length = len(text)
        text = emoji.replace_emoji(text, replace='')
        if len(text) < original_length:
            # Emoji were removed - not necessarily an error, just informational
            pass
    except ImportError:
        # emoji package not installed - skip emoji removal
        pass
    
    # Check minimum length after cleaning
    stripped = text.strip()
    if len(stripped) < 10:
        raise ValidationError(
            f"Text too short after cleaning: {len(stripped)} characters (minimum 10). "
            f"Provide more substantial text content."
        )
    
    # Check for only special characters
    alphanumeric_count = sum(c.isalnum() for c in stripped)
    if alphanumeric_count < 5:
        raise ValidationError(
            f"Text contains too few alphanumeric characters ({alphanumeric_count}). "
            f"Provide readable text content."
        )
    
    return text


def validate_reference_audio(audio_path: Path) -> dict:
    """
    Validate reference audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with audio metadata
        
    Raises:
        ValidationError: If audio is invalid
    """
    if not audio_path.exists():
        raise ValidationError(f"Audio file not found: {audio_path}")
    
    if not audio_path.is_file():
        raise ValidationError(f"Path is not a file: {audio_path}")
    
    # Check file size (max 50MB)
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 50:
        raise ValidationError(
            f"Audio file too large: {file_size_mb:.1f}MB (maximum 50MB). "
            f"Use a shorter reference clip."
        )
    
    # Try to load audio
    try:
        audio, sr = librosa.load(str(audio_path), sr=None, mono=False)
    except Exception as e:
        raise ValidationError(f"Failed to load audio file: {e}")
    
    # Check duration
    duration = librosa.get_duration(y=audio, sr=sr)
    if duration < 5.0:
        raise ValidationError(
            f"Audio too short: {duration:.1f}s (minimum 5 seconds). "
            f"Provide a longer reference sample."
        )
    
    if duration > 120.0:
        raise ValidationError(
            f"Audio too long: {duration:.1f}s (maximum 120 seconds). "
            f"Use a shorter clip for better performance."
        )
    
    # Check for silence
    if len(audio.shape) > 1:
        audio_mono = librosa.to_mono(audio)
    else:
        audio_mono = audio
    
    rms = librosa.feature.rms(y=audio_mono)[0]
    mean_rms = float(np.mean(rms))
    
    if mean_rms < 0.001:
        raise ValidationError(
            f"Audio appears to be silent (RMS: {mean_rms:.6f}). "
            f"Provide audio with clear voice content."
        )
    
    # Check sample rate
    if sr < 8000:
        raise ValidationError(
            f"Sample rate too low: {sr}Hz (minimum 8000Hz). "
            f"Use higher quality audio."
        )
    
    if sr > 48000:
        raise ValidationError(
            f"Sample rate too high: {sr}Hz (maximum 48000Hz). "
            f"Audio will be resampled which may affect quality."
        )
    
    return {
        "duration_seconds": duration,
        "sample_rate": sr,
        "channels": "stereo" if len(audio.shape) > 1 else "mono",
        "rms_level": mean_rms,
        "file_size_mb": file_size_mb
    }
