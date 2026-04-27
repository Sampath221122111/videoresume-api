"""
Transcription Service — uses Groq's free Whisper API.

Why Groq over local Whisper:
- Render free tier has 512MB RAM — local whisper-small needs ~1GB
- Groq provides whisper-large-v3 for FREE with generous limits
- Much faster than local inference
- Returns timestamped segments (critical for highlight selection)
"""

from groq import Groq
from app.config import get_settings
from app.models.schemas import TranscriptionResult, TranscriptSegment


def transcribe_audio(audio_path: str) -> TranscriptionResult:
    """
    Transcribe audio file using Groq Whisper API.
    Returns full text + timestamped segments.
    """
    settings = get_settings()
    client = Groq(api_key=settings.groq_api_key)

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=("audio.wav", audio_file),
            model="whisper-large-v3",
            response_format="verbose_json",
            language="en",
            timestamp_granularities=["segment"],
        )

    # Parse segments
    segments = []
    if hasattr(response, "segments") and response.segments:
        for seg in response.segments:
            segments.append(TranscriptSegment(
                start=seg.get("start", seg.get("start", 0)),
                end=seg.get("end", seg.get("end", 0)),
                text=seg.get("text", "").strip(),
            ))

    # Calculate duration from last segment
    duration = segments[-1].end if segments else 0.0

    return TranscriptionResult(
        full_text=response.text.strip() if hasattr(response, "text") else "",
        segments=segments,
        language="en",
        duration=duration,
    )
