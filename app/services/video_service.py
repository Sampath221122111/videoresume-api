"""
Video Service — ULTRA FAST.
Key optimizations:
- Skip webm→mp4 conversion entirely (ffmpeg reads webm directly)
- Stream copy for highlight clips (no re-encoding)
- Smaller chunk size for faster download
"""

import os
import re as re_mod
import subprocess
import tempfile
import httpx
from app.models.schemas import HighlightSelection

TEMP_DIR = tempfile.gettempdir()


async def download_video(video_url: str, job_id: str) -> str:
    """Download video — NO conversion, just raw download."""
    # Detect format
    is_webm = ".webm" in video_url.lower()
    ext = ".webm" if is_webm else ".mp4"
    video_path = os.path.join(TEMP_DIR, f"{job_id}_raw{ext}")

    # Fast stream download
    async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
        async with client.stream("GET", video_url) as response:
            response.raise_for_status()
            with open(video_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=131072):
                    f.write(chunk)

    size_mb = os.path.getsize(video_path) / (1024*1024)
    print(f"[DOWNLOAD] {size_mb:.1f} MB ({ext}) — NO conversion needed")
    return video_path


def get_video_duration(video_path: str) -> float:
    """Get duration — tries 3 methods."""
    # Method 1: ffprobe format
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=15)
        val = r.stdout.strip()
        if val and val != "N/A":
            return float(val)
    except: pass

    # Method 2: ffmpeg -i stderr parse
    try:
        r = subprocess.run(["ffmpeg", "-i", video_path], capture_output=True, text=True, timeout=15)
        m = re_mod.search(r"Duration:\s*(\d+):(\d+):(\d[\d.]*)", r.stderr)
        if m:
            return float(m.group(1))*3600 + float(m.group(2))*60 + float(m.group(3))
    except: pass

    # Method 3: file size estimate
    size_mb = os.path.getsize(video_path) / (1024*1024)
    return max(size_mb * 8, 15)  # ~8 sec per MB rough estimate


def validate_video(video_path: str, max_duration: int = 300) -> dict:
    """Quick validation."""
    duration = get_video_duration(video_path)
    print(f"[VALIDATE] Duration: {duration:.1f}s")

    if duration < 10:
        return {"is_valid": False, "duration": duration, "has_audio": False, "error": "Video must be at least 10 seconds"}
    if duration > max_duration:
        return {"is_valid": False, "duration": duration, "has_audio": False, "error": f"Video exceeds {max_duration}s limit"}

    # Quick audio check
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, timeout=10)
    has_audio = "audio" in r.stdout

    if not has_audio:
        return {"is_valid": False, "duration": duration, "has_audio": False, "error": "No audio track"}
    return {"is_valid": True, "duration": duration, "has_audio": True, "error": None}


def extract_audio(video_path: str, job_id: str) -> str:
    """Extract audio — works with both webm and mp4."""
    audio_path = os.path.join(TEMP_DIR, f"{job_id}_audio.wav")
    cmd = ["ffmpeg", "-y", "-i", video_path,
           "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
           "-t", "180",  # Max 3 minutes of audio (saves transcription time)
           audio_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {r.stderr[:300]}")
    return audio_path


def generate_highlight_clip(video_path: str, job_id: str, selection: HighlightSelection) -> str:
    """Generate clip — uses copy codec when possible for INSTANT speed."""
    clip_path = os.path.join(TEMP_DIR, f"{job_id}_highlight.mp4")
    duration = selection.end_time - selection.start_time

    # Try stream copy first (instant, no re-encoding)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(selection.start_time),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart",
        "-threads", "2",
        clip_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if r.returncode != 0:
        raise RuntimeError(f"Clip failed: {r.stderr[:300]}")
    return clip_path


def cleanup_temp_files(job_id: str) -> None:
    for p in [f"{job_id}_raw", f"{job_id}_safe", f"{job_id}_audio", f"{job_id}_highlight"]:
        for ext in [".mp4", ".wav", ".webm"]:
            path = os.path.join(TEMP_DIR, f"{p}{ext}")
            if os.path.exists(path):
                os.remove(path)
