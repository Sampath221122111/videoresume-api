"""
Video Service — handles all FFmpeg operations.
OPTIMIZED: Fast download, handles webm, robust duration detection.
"""

import os
import re as re_mod
import subprocess
import tempfile
import httpx
from app.models.schemas import HighlightSelection

TEMP_DIR = tempfile.gettempdir()


async def download_video(video_url: str, job_id: str) -> str:
    """
    Download video and ensure it's in a processable format.
    Handles .webm from browser recording by converting to .mp4.
    """
    # Detect format from URL
    is_webm = ".webm" in video_url.lower()
    raw_ext = ".webm" if is_webm else ".mp4"
    raw_path = os.path.join(TEMP_DIR, f"{job_id}_raw{raw_ext}")
    final_path = os.path.join(TEMP_DIR, f"{job_id}_safe.mp4")

    # Stream download
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        async with client.stream("GET", video_url) as response:
            response.raise_for_status()
            with open(raw_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    f.write(chunk)

    file_size = os.path.getsize(raw_path)
    print(f"[DOWNLOAD] Video downloaded: {file_size / (1024*1024):.1f} MB ({raw_ext})")

    # If webm or duration can't be read, convert to mp4 with fast preset
    if is_webm or not _can_read_duration(raw_path):
        print(f"[DOWNLOAD] Converting to mp4 for compatibility...")
        cmd = [
            "ffmpeg", "-y", "-i", raw_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            "-max_muxing_queue_size", "2048",
            final_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"[DOWNLOAD] FFmpeg stderr: {result.stderr[:300]}")
            # If conversion fails, try using raw file directly
            if os.path.exists(raw_path):
                os.rename(raw_path, final_path)
        else:
            # Clean up raw
            if os.path.exists(raw_path):
                os.remove(raw_path)
    else:
        # mp4 with readable duration — just rename
        os.rename(raw_path, final_path)

    return final_path


def _can_read_duration(path: str) -> bool:
    """Check if we can read duration from this file."""
    try:
        dur = get_video_duration(path)
        return dur > 0
    except Exception:
        return False


def get_video_duration(video_path: str) -> float:
    """
    Get video duration using multiple methods.
    Handles webm, mp4, and other formats robustly.
    """
    # Method 1: format duration
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        val = r.stdout.strip()
        if val and val != "N/A":
            d = float(val)
            if d > 0:
                return d
    except Exception:
        pass

    # Method 2: stream duration
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        val = r.stdout.strip()
        if val and val != "N/A":
            d = float(val)
            if d > 0:
                return d
    except Exception:
        pass

    # Method 3: Parse duration from ffmpeg -i output (works for ALL formats)
    try:
        cmd = ["ffmpeg", "-i", video_path]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        # Duration is in stderr: "Duration: 00:02:34.56"
        match = re_mod.search(r"Duration:\s*(\d+):(\d+):(\d[\d.]*)", r.stderr)
        if match:
            h, m, s = float(match.group(1)), float(match.group(2)), float(match.group(3))
            d = h * 3600 + m * 60 + s
            if d > 0:
                return d
    except Exception:
        pass

    # Method 4: Estimate from file size (last resort — assume ~1MB per 10 seconds)
    try:
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        estimated = size_mb * 10  # rough estimate
        print(f"[WARN] Duration estimated from file size: ~{estimated:.0f}s")
        if estimated > 5:
            return estimated
    except Exception:
        pass

    raise RuntimeError("Cannot determine video duration")


def validate_video(video_path: str, max_duration: int = 300) -> dict:
    """Validate video meets requirements."""
    duration = get_video_duration(video_path)
    print(f"[VALIDATE] Video duration: {duration:.1f}s")

    # Check audio track
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a",
           "-show_entries", "stream=codec_type",
           "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    audio_result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    has_audio = "audio" in audio_result.stdout

    if duration < 10:  # Lowered from 30 to 10 seconds minimum
        return {"is_valid": False, "duration": duration, "has_audio": has_audio,
                "error": "Video must be at least 10 seconds"}
    if duration > max_duration:
        return {"is_valid": False, "duration": duration, "has_audio": has_audio,
                "error": f"Video exceeds {max_duration}s limit"}
    if not has_audio:
        return {"is_valid": False, "duration": duration, "has_audio": False,
                "error": "Video has no audio track"}

    return {"is_valid": True, "duration": duration, "has_audio": True, "error": None}


def extract_audio(video_path: str, job_id: str) -> str:
    """Extract audio as 16kHz mono WAV."""
    audio_path = os.path.join(TEMP_DIR, f"{job_id}_audio.wav")
    cmd = ["ffmpeg", "-y", "-i", video_path,
           "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
           audio_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr[:500]}")
    return audio_path


def generate_highlight_clip(video_path: str, job_id: str, selection: HighlightSelection) -> str:
    """Cut a highlight clip. Uses ultrafast preset for speed."""
    clip_path = os.path.join(TEMP_DIR, f"{job_id}_highlight.mp4")
    duration = selection.end_time - selection.start_time

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(selection.start_time),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-vf", f"fade=t=in:st=0:d=0.5,fade=t=out:st={max(0, duration - 0.5)}:d=0.5",
        "-af", f"afade=t=in:st=0:d=0.5,afade=t=out:st={max(0, duration - 0.5)}:d=0.5",
        "-movflags", "+faststart",
        clip_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Highlight clip generation failed: {result.stderr[:500]}")
    return clip_path


def cleanup_temp_files(job_id: str) -> None:
    """Remove all temporary files for a job."""
    for pattern in [f"{job_id}_raw", f"{job_id}_safe", f"{job_id}_audio", f"{job_id}_highlight"]:
        for ext in [".mp4", ".wav", ".webm"]:
            path = os.path.join(TEMP_DIR, f"{pattern}{ext}")
            if os.path.exists(path):
                os.remove(path)
