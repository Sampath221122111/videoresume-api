"""
Tone Analysis Service — uses librosa for audio feature extraction.

Analyzes:
- Energy (RMS) — how confidently the person speaks
- Pitch (F0) — vocal stability and variation
- Speaking rate — words per minute from transcript
- Per-second energy timeline — used for highlight clip selection

Output: confidence_score (0-100) combining all factors.
"""

import numpy as np
import librosa
from app.models.schemas import AudioAnalysis, TranscriptionResult


def analyze_tone(audio_path: str, transcript: TranscriptionResult) -> AudioAnalysis:
    """
    Full tone analysis pipeline.
    Returns structured AudioAnalysis with per-second energy for highlight selection.
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    if duration < 1:
        return AudioAnalysis()

    # --- Energy (RMS) ---
    # Per-frame RMS energy
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    avg_energy = float(np.mean(rms))
    energy_std = float(np.std(rms))

    # Per-second energy timeline (for highlight clip selection)
    frames_per_second = sr // 512
    energy_timeline = []
    for i in range(0, len(rms), max(1, frames_per_second)):
        chunk = rms[i:i + frames_per_second]
        if len(chunk) > 0:
            energy_timeline.append(float(np.mean(chunk)))

    # --- Pitch (F0) ---
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr
    )
    # Filter NaN (unvoiced frames)
    f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    pitch_mean = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0
    pitch_std = float(np.std(f0_clean)) if len(f0_clean) > 0 else 0.0

    # --- Speaking Rate ---
    word_count = len(transcript.full_text.split()) if transcript.full_text else 0
    speaking_rate = (word_count / duration) * 60 if duration > 0 else 0  # WPM

    # --- Confidence Score (0-100) ---
    # Combine factors with weights:
    # - Energy: higher = more confident (weight: 35%)
    # - Pitch stability: lower std = more confident (weight: 25%)
    # - Speaking rate: 120-160 WPM is ideal (weight: 20%)
    # - Energy consistency: lower std = more steady (weight: 20%)

    # Normalize energy (0-100 scale, assuming typical range)
    energy_score = min(100, (avg_energy / 0.05) * 100) if avg_energy > 0 else 0

    # Pitch stability (lower variation = higher score)
    pitch_cv = (pitch_std / pitch_mean) if pitch_mean > 0 else 1.0
    pitch_score = max(0, min(100, (1 - pitch_cv) * 100))

    # Speaking rate score (ideal: 120-160 WPM)
    if 120 <= speaking_rate <= 160:
        rate_score = 100
    elif speaking_rate < 120:
        rate_score = max(0, (speaking_rate / 120) * 100)
    else:
        rate_score = max(0, 100 - ((speaking_rate - 160) / 40) * 100)

    # Energy consistency
    energy_cv = (energy_std / avg_energy) if avg_energy > 0 else 1.0
    consistency_score = max(0, min(100, (1 - energy_cv) * 100))

    confidence_score = (
        energy_score * 0.35 +
        pitch_score * 0.25 +
        rate_score * 0.20 +
        consistency_score * 0.20
    )

    return AudioAnalysis(
        average_energy=round(avg_energy, 6),
        energy_std=round(energy_std, 6),
        speaking_rate_wpm=round(speaking_rate, 1),
        pitch_mean=round(pitch_mean, 2),
        pitch_std=round(pitch_std, 2),
        confidence_score=round(min(100, max(0, confidence_score)), 1),
        energy_timeline=[round(e, 6) for e in energy_timeline],
    )
