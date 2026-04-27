"""
Job Queue Manager — uses Upstash Redis for job tracking.

Why Redis over in-memory:
- Survives server restarts on Render free tier (which sleeps after 15min)
- Allows frontend to poll job status independently
- Rate limits: tracks per-user job count

Job lifecycle: QUEUED → [processing stages] → COMPLETED / FAILED
"""

import json
import redis
from typing import Optional
from app.config import get_settings
from app.models.schemas import JobStatus, JobStatusResponse, ProcessingResult


def get_redis() -> redis.Redis:
    settings = get_settings()
    return redis.from_url(settings.upstash_redis_url, decode_responses=True)


class JobQueue:

    JOB_PREFIX = "job:"
    USER_JOBS_PREFIX = "user_jobs:"
    JOB_TTL = 86400  # 24 hours

    @staticmethod
    def create_job(job_id: str, user_id: str, submission_id: str) -> None:
        """Create a new processing job."""
        r = get_redis()
        job_data = {
            "job_id": job_id,
            "user_id": user_id,
            "submission_id": submission_id,
            "status": JobStatus.QUEUED.value,
            "progress": 0,
            "message": "Job queued for processing",
            "result": None,
            "error": None,
        }
        r.setex(f"{JobQueue.JOB_PREFIX}{job_id}", JobQueue.JOB_TTL, json.dumps(job_data))
        # Track per-user jobs for rate limiting
        r.sadd(f"{JobQueue.USER_JOBS_PREFIX}{user_id}", job_id)
        r.expire(f"{JobQueue.USER_JOBS_PREFIX}{user_id}", JobQueue.JOB_TTL)

    @staticmethod
    def update_job(job_id: str, status: JobStatus, progress: int, message: str = "") -> None:
        """Update job status and progress."""
        r = get_redis()
        key = f"{JobQueue.JOB_PREFIX}{job_id}"
        raw = r.get(key)
        if not raw:
            return
        data = json.loads(raw)
        data["status"] = status.value
        data["progress"] = progress
        data["message"] = message
        r.setex(key, JobQueue.JOB_TTL, json.dumps(data))

    @staticmethod
    def complete_job(job_id: str, result: ProcessingResult) -> None:
        """Mark job as completed with results."""
        r = get_redis()
        key = f"{JobQueue.JOB_PREFIX}{job_id}"
        raw = r.get(key)
        if not raw:
            return
        data = json.loads(raw)
        data["status"] = JobStatus.COMPLETED.value
        data["progress"] = 100
        data["message"] = "Processing complete"
        data["result"] = result.model_dump()
        r.setex(key, JobQueue.JOB_TTL, json.dumps(data))

    @staticmethod
    def fail_job(job_id: str, error: str) -> None:
        """Mark job as failed."""
        r = get_redis()
        key = f"{JobQueue.JOB_PREFIX}{job_id}"
        raw = r.get(key)
        if not raw:
            return
        data = json.loads(raw)
        data["status"] = JobStatus.FAILED.value
        data["message"] = "Processing failed"
        data["error"] = error
        r.setex(key, JobQueue.JOB_TTL, json.dumps(data))

    @staticmethod
    def get_job(job_id: str) -> Optional[JobStatusResponse]:
        """Get current job status."""
        r = get_redis()
        raw = r.get(f"{JobQueue.JOB_PREFIX}{job_id}")
        if not raw:
            return None
        data = json.loads(raw)
        result = None
        if data.get("result"):
            result = ProcessingResult(**data["result"])
        return JobStatusResponse(
            job_id=data["job_id"],
            status=JobStatus(data["status"]),
            progress=data.get("progress", 0),
            message=data.get("message", ""),
            result=result,
            error=data.get("error"),
        )

    @staticmethod
    def get_user_job_count(user_id: str) -> int:
        """Count active jobs for rate limiting."""
        r = get_redis()
        job_ids = r.smembers(f"{JobQueue.USER_JOBS_PREFIX}{user_id}")
        active = 0
        for jid in job_ids:
            raw = r.get(f"{JobQueue.JOB_PREFIX}{jid}")
            if raw:
                data = json.loads(raw)
                if data["status"] not in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
                    active += 1
        return active

    @staticmethod
    def get_user_total_submissions(user_id: str) -> int:
        """Total submissions (completed + active) for free tier cap."""
        r = get_redis()
        return r.scard(f"{JobQueue.USER_JOBS_PREFIX}{user_id}") or 0
