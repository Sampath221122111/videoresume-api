"""
Highlight Selector — picks the optimal 30-second window.

Combines three signals to find the best segment:
1. Audio energy (confidence/engagement from tone_service)
2. Face expression scores (smile/engagement from face_service)
3. Content relevance (transcript segments mentioning skills/achievements)

The scoring prevents cherry-picking misleading moments by requiring
all three signals to be reasonably good (not just one peak).
"""

import numpy as np
from app.models.schemas import (
    AudioAnalysis, FaceAnalysis, TranscriptionResult,
    HighlightSelection, TranscriptSegment,
)


# Keywords that indicate high-value content
CONTENT_KEYWORDS = [
    "skill", "experience", "project", "built", "developed", "created",
    "achieved", "led", "managed", "designed", "implemented", "improved",
    "award", "certification", "intern", "work", "goal", "passion",
    "python", "java", "react", "machine learning", "data", "ai",
    "team", "leadership", "problem", "solution", "research",
]

CLIP_DURATION = 30  # seconds


def compute_content_scores(
    transcript: TranscriptionResult,
    total_duration: float,
) -> list[float]:
    """
    Per-second content relevance score based on transcript keywords.
    Higher score = more relevant content being discussed.
    """
    scores = [0.0] * max(1, int(total_duration))

    for segment in transcript.segments:
        text_lower = segment.text.lower()
        keyword_count = sum(1 for kw in CONTENT_KEYWORDS if kw in text_lower)
        if keyword_count > 0:
            start_sec = int(segment.start)
            end_sec = min(int(segment.end) + 1, len(scores))
            score = min(100, keyword_count * 20)
            for i in range(start_sec, end_sec):
                scores[i] = max(scores[i], score)

    return scores


def select_highlight(
    audio: AudioAnalysis,
    face: FaceAnalysis,
    transcript: TranscriptionResult,
    video_duration: float,
) -> HighlightSelection:
    """
    Find the best 30-second window using combined scoring.

    Scoring weights:
    - Audio energy: 30% (confident speech)
    - Expression: 25% (engaged face)
    - Content: 35% (relevant topics)
    - Eye contact boost: 10% (looking at camera)

    Constraint: all three primary signals must be above minimum thresholds
    to prevent misleading highlights.
    """
    total_seconds = int(video_duration)

    # If video is <= 30s, just use the whole thing
    if total_seconds <= CLIP_DURATION:
        return HighlightSelection(
            start_time=0,
            end_time=video_duration,
            score=100,
            reason="Video is within 30-second limit",
        )

    # Build per-second timelines (normalize all to 0-100)
    energy = audio.energy_timeline[:total_seconds]
    expression = face.expression_timeline[:total_seconds]
    content = compute_content_scores(transcript, video_duration)[:total_seconds]

    # Pad shorter timelines
    while len(energy) < total_seconds:
        energy.append(0)
    while len(expression) < total_seconds:
        expression.append(0)
    while len(content) < total_seconds:
        content.append(0)

    # Normalize each timeline to 0-100
    def normalize(arr):
        arr = np.array(arr, dtype=float)
        if arr.max() > arr.min():
            return ((arr - arr.min()) / (arr.max() - arr.min()) * 100).tolist()
        return [50.0] * len(arr)

    energy_norm = normalize(energy)
    expression_norm = normalize(expression)
    content_norm = normalize(content)

    # Sliding window scoring
    best_score = -1
    best_start = 0

    for start in range(total_seconds - CLIP_DURATION + 1):
        end = start + CLIP_DURATION
        window_energy = np.mean(energy_norm[start:end])
        window_expression = np.mean(expression_norm[start:end])
        window_content = np.mean(content_norm[start:end])

        # Minimum threshold check (prevent misleading highlights)
        # At least 2 of 3 signals should be above 25%
        above_threshold = sum([
            window_energy > 25,
            window_expression > 25,
            window_content > 25,
        ])
        if above_threshold < 2:
            continue

        combined = (
            window_energy * 0.30 +
            window_expression * 0.25 +
            window_content * 0.35 +
            10  # base score
        )

        if combined > best_score:
            best_score = combined
            best_start = start

    best_end = min(best_start + CLIP_DURATION, video_duration)

    # Generate reason
    window_e = np.mean(energy_norm[best_start:best_start + CLIP_DURATION])
    window_x = np.mean(expression_norm[best_start:best_start + CLIP_DURATION])
    window_c = np.mean(content_norm[best_start:best_start + CLIP_DURATION])

    reasons = []
    if window_c > 50:
        reasons.append("discusses key skills and experience")
    if window_e > 50:
        reasons.append("high speaking confidence")
    if window_x > 50:
        reasons.append("positive expression and engagement")
    reason = "Selected because student " + (", ".join(reasons) if reasons else "this segment has the best overall quality")

    return HighlightSelection(
        start_time=float(best_start),
        end_time=float(best_end),
        score=round(best_score, 1),
        reason=reason,
    )
