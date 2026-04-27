"""
Cloudinary upload service — uploads resume PDFs and highlight clips.
Uses PUBLIC access so users can view/download without authentication.
"""

import cloudinary
import cloudinary.uploader
from app.config import get_settings


def get_cloudinary():
    settings = get_settings()
    cloudinary.config(
        cloud_name=settings.cloudinary_cloud_name,
        api_key=settings.cloudinary_api_key,
        api_secret=settings.cloudinary_api_secret,
        secure=True,
    )


def upload_pdf(pdf_path: str, user_id: str, job_id: str) -> str:
    """Upload resume PDF to Cloudinary. Returns public URL."""
    get_cloudinary()

    result = cloudinary.uploader.upload(
        pdf_path,
        resource_type="image",
        type="upload",
        folder=f"video-resume/{user_id}/resumes",
        public_id=f"resume_{job_id}",
        format="pdf",
        access_mode="public",
    )

    print(f"[CLOUDINARY] PDF uploaded: {result.get('secure_url', 'NO URL')}")
    return result["secure_url"]


def upload_video_clip(clip_path: str, user_id: str, job_id: str) -> str:
    """Upload highlight clip to Cloudinary. Returns public URL."""
    get_cloudinary()

    result = cloudinary.uploader.upload(
        clip_path,
        resource_type="video",
        type="upload",
        folder=f"video-resume/{user_id}/clips",
        public_id=f"clip_{job_id}",
        access_mode="public",
    )

    print(f"[CLOUDINARY] Clip uploaded: {result.get('secure_url', 'NO URL')}")
    return result["secure_url"]
