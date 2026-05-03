"""
API Router — FIXED: Force-clears Redis before checking jobs.
No more stuck job blocking.
"""

import uuid
import re
import json
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from app.models.schemas import (
    ProcessVideoRequest, ProcessVideoResponse,
    JobStatusResponse, JobStatus, HealthResponse,
)
from app.utils.auth import verify_token
from app.utils.queue import JobQueue, get_redis
from app.services.pipeline import run_pipeline
from app.config import get_settings

router = APIRouter()


@router.post("/api/process-video", response_model=ProcessVideoResponse)
async def process_video(
    request: ProcessVideoRequest,
    background_tasks: BackgroundTasks,
    auth: dict = Depends(verify_token),
):
    user_id = auth["user_id"]

    if request.user_id != user_id:
        raise HTTPException(status_code=403, detail="Cannot process videos for other users")

    if not re.match(r"https?://res\.cloudinary\.com/", request.video_url):
        raise HTTPException(status_code=400, detail="Invalid video URL")

    # FORCE clear ALL old jobs for this user — no more stuck blocking
    _force_clear_user_jobs(user_id)

    # Now create fresh job
    job_id = str(uuid.uuid4())
    JobQueue.create_job(job_id, user_id, request.submission_id)
    background_tasks.add_task(run_pipeline, job_id, request)

    print(f"[OK] Started job {job_id[:8]} for user {user_id[:8]}")
    return ProcessVideoResponse(job_id=job_id, status=JobStatus.QUEUED, message="Processing started.")


def _force_clear_user_jobs(user_id: str):
    """Force clear ALL Redis jobs for this user — prevents any blocking."""
    try:
        r = get_redis()
        key = f"{JobQueue.USER_JOBS_PREFIX}{user_id}"
        job_ids = r.smembers(key)
        count = len(job_ids)
        for jid in job_ids:
            r.delete(f"{JobQueue.JOB_PREFIX}{jid}")
        r.delete(key)
        if count > 0:
            print(f"[FORCE-CLEAR] Cleared {count} old jobs for user {user_id[:8]}")
    except Exception as e:
        print(f"[FORCE-CLEAR] Redis error (ignoring): {e}")


@router.post("/api/clear-jobs")
async def clear_jobs(auth: dict = Depends(verify_token)):
    user_id = auth["user_id"]
    _force_clear_user_jobs(user_id)
    return {"cleared": "all"}


@router.get("/api/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, auth: dict = Depends(verify_token)):
    job = JobQueue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/api/submissions")
async def get_submissions(auth: dict = Depends(verify_token)):
    from app.services.supabase_service import get_supabase_client
    client = get_supabase_client()
    result = client.table("submissions").select("*").eq(
        "user_id", auth["user_id"]
    ).order("created_at", desc=True).execute()
    return {"submissions": result.data or []}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    services = {}
    try:
        r = get_redis(); r.ping(); services["redis"] = "connected"
    except Exception as e:
        services["redis"] = f"error: {str(e)[:50]}"
    try:
        from app.services.supabase_service import get_supabase_client
        get_supabase_client().table("profiles").select("id").limit(1).execute()
        services["supabase"] = "connected"
    except Exception as e:
        services["supabase"] = f"error: {str(e)[:50]}"
    try:
        from groq import Groq
        Groq(api_key=get_settings().groq_api_key).models.list()
        services["groq"] = "connected"
    except Exception as e:
        services["groq"] = f"error: {str(e)[:50]}"
    return HealthResponse(
        status="healthy" if all(v == "connected" for v in services.values()) else "degraded",
        version="1.0.0", services=services
    )
