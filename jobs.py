# File: jobs.py
# Job queue management for async TTS processing
# Handles long-running synthesis jobs that exceed Modal's 10-minute timeout

import uuid
import logging
import copy
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job status states"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobInfo:
    """Information about a TTS job"""
    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Request details
    text_length: int = 0
    chunk_count: int = 0
    processed_chunks: int = 0
    
    # Result details
    result_file: Optional[str] = None
    error_message: Optional[str] = None
    
    # Progress tracking
    progress_percent: float = 0.0
    current_step: str = "Queued"
    
    # Failed chunks for partial success
    failed_chunks: list = field(default_factory=list)


class JobManager:
    """
    Manages async TTS jobs.
    
    Uses in-memory storage (could be replaced with Redis for production).
    """
    
    def __init__(self):
        self._jobs: Dict[str, JobInfo] = {}
        self._lock = threading.Lock()
        logger.info("JobManager initialized")
    
    def create_job(self, text_length: int) -> str:
        """
        Create a new job and return its ID.
        
        Args:
            text_length: Length of input text
            
        Returns:
            Job ID (UUID)
        """
        job_id = str(uuid.uuid4())
        
        with self._lock:
            self._jobs[job_id] = JobInfo(
                job_id=job_id,
                status=JobStatus.QUEUED,
                created_at=datetime.now(),
                text_length=text_length
            )
        
        logger.info(f"Created job {job_id} for text length {text_length}")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[JobInfo]:
        """Get job info by ID (returns a deep copy to prevent external modification)"""
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job else None
    
    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        **kwargs
    ) -> bool:
        """
        Update job status and optional fields.
        
        Args:
            job_id: Job ID
            status: New status
            **kwargs: Additional fields to update (error_message, result_file, etc.)
            
        Returns:
            True if updated, False if job not found
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                logger.warning(f"Job {job_id} not found for status update")
                return False
            
            job.status = status
            
            # Update timestamp based on status
            if status == JobStatus.PROCESSING and not job.started_at:
                job.started_at = datetime.now()
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                job.completed_at = datetime.now()
            
            # Update optional fields
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            
            logger.info(f"Job {job_id} status updated to {status}")
            return True
    
    def update_progress(
        self,
        job_id: str,
        processed_chunks: int,
        chunk_count: int,
        current_step: str
    ):
        """Update job progress"""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            
            job.processed_chunks = processed_chunks
            job.chunk_count = chunk_count
            job.current_step = current_step
            
            if chunk_count > 0:
                job.progress_percent = (processed_chunks / chunk_count) * 100
            
            logger.debug(
                f"Job {job_id} progress: {processed_chunks}/{chunk_count} "
                f"({job.progress_percent:.1f}%) - {current_step}"
            )
    
    def add_failed_chunk(self, job_id: str, chunk_info: Dict[str, Any]):
        """Record a failed chunk"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.failed_chunks.append(chunk_info)
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Remove jobs older than specified hours"""
        cutoff = datetime.now()
        count = 0
        
        with self._lock:
            job_ids_to_remove = []
            
            for job_id, job in self._jobs.items():
                age_hours = (cutoff - job.created_at).total_seconds() / 3600
                if age_hours > max_age_hours:
                    job_ids_to_remove.append(job_id)
                    # Also delete result file if exists
                    if job.result_file:
                        try:
                            Path(job.result_file).unlink(missing_ok=True)
                        except Exception as e:
                            logger.error(f"Failed to delete result file {job.result_file}: {e}")
            
            for job_id in job_ids_to_remove:
                del self._jobs[job_id]
                count += 1
        
        if count > 0:
            logger.info(f"Cleaned up {count} old jobs (>{max_age_hours}h)")
        
        return count
    
    def get_all_jobs(self) -> Dict[str, JobInfo]:
        """Get all jobs (for debugging/admin)"""
        with self._lock:
            return dict(self._jobs)


# Global instance
_job_manager: Optional[JobManager] = None
_job_manager_lock = threading.Lock()


def get_job_manager() -> JobManager:
    """Get or create the global job manager instance (thread-safe)"""
    global _job_manager
    if _job_manager is None:
        with _job_manager_lock:
            # Double-check locking pattern to prevent race conditions
            if _job_manager is None:
                _job_manager = JobManager()
    return _job_manager
