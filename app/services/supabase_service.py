"""
Supabase Service — updates the submissions table with processing results.

Uses the service role key (not anon key) because this runs server-side
and needs to bypass Row Level Security to update records.

Operations:
- Update submission status (processing → completed / failed)
- Store resume URL, clip URL, analysis scores
- Store extracted skills for searchability
"""

from supabase import create_client
from app.config import get_settings
from app.models.schemas import ProcessingResult


def get_supabase_client():
    """Get Supabase client with service role key (bypasses RLS)."""
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_service_key)


def update_submission_processing(submission_id: str) -> None:
    """Mark submission as processing."""
    client = get_supabase_client()
    client.table("submissions").update({
        "status": "processing",
    }).eq("id", submission_id).execute()


def update_submission_completed(submission_id: str, result: ProcessingResult) -> None:
    """Update submission with completed results."""
    client = get_supabase_client()

    # Update main submission record
    client.table("submissions").update({
        "status": "completed",
        "transcript": result.transcript,
        "resume_pdf_url": result.resume_pdf_url,
        "highlight_clip_url": result.highlight_clip_url,
        "confidence_score": result.confidence_score,
        "energy_score": result.energy_score,
        "expression_score": result.expression_score,
    }).eq("id", submission_id).execute()

    # Insert extracted skills (for searchability)
    if result.skills_extracted:
        skills_rows = [
            {"submission_id": submission_id, "skill_name": skill}
            for skill in result.skills_extracted
        ]
        client.table("extracted_skills").upsert(skills_rows).execute()


def update_submission_failed(submission_id: str, error: str) -> None:
    """Mark submission as failed."""
    client = get_supabase_client()
    client.table("submissions").update({
        "status": "failed",
    }).eq("id", submission_id).execute()


def get_user_submission_count(user_id: str) -> int:
    """Count total submissions for free tier cap enforcement."""
    client = get_supabase_client()
    result = client.table("submissions").select(
        "id", count="exact"
    ).eq("user_id", user_id).execute()
    return result.count or 0
