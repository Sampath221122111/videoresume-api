"""
Pipeline — ULTRA MINIMAL for 512MB RAM.
NO librosa, NO mediapipe, NO opencv loaded.
Only: ffmpeg (system) + Groq API (cloud) + ReportLab (small).
All scoring estimated from transcript quality.
"""

import time
import traceback
import os
from concurrent.futures import ThreadPoolExecutor
from app.models.schemas import (
    JobStatus, ProcessingResult, ProcessVideoRequest,
    FaceAnalysis, AudioAnalysis, TranscriptionResult,
)
from app.utils.queue import JobQueue
from app.services import (
    download_video, validate_video, extract_audio,
    transcribe_audio,
    select_highlight, generate_resume, generate_pdf,
    generate_highlight_clip, cleanup_temp_files,
    upload_pdf, upload_video_clip,
    update_submission_processing, update_submission_completed,
    update_submission_failed,
)


async def run_pipeline(job_id: str, request: ProcessVideoRequest) -> None:
    video_path = None
    t0 = time.time()

    def log(stage, start):
        print(f"[PIPELINE] {stage}: {time.time()-start:.1f}s")

    try:
        # ── Download ──
        t = time.time()
        JobQueue.update_job(job_id, JobStatus.DOWNLOADING, 5, "Downloading video...")
        update_submission_processing(request.submission_id)
        video_path = await download_video(request.video_url, job_id)
        log("Download", t)

        # ── Validate ──
        t = time.time()
        validation = validate_video(video_path)
        if not validation["is_valid"]:
            raise ValueError(f"Video validation failed: {validation['error']}")
        video_duration = validation["duration"]
        log("Validate", t)

        # ── Extract Audio ──
        t = time.time()
        JobQueue.update_job(job_id, JobStatus.EXTRACTING_AUDIO, 15, "Extracting audio...")
        audio_path = extract_audio(video_path, job_id)
        log("Extract Audio", t)

        # ── Transcribe via Groq Cloud API (no local memory) ──
        t = time.time()
        JobQueue.update_job(job_id, JobStatus.TRANSCRIBING, 30, "Transcribing speech...")
        transcript = transcribe_audio(audio_path)
        log("Transcribe", t)

        if not transcript.full_text.strip():
            raise ValueError("No speech detected. Record with clear audio.")

        # ── Score from transcript (NO librosa, NO mediapipe) ──
        JobQueue.update_job(job_id, JobStatus.ANALYZING_TONE, 45, "Analyzing performance...")
        audio_analysis = _score_from_transcript(transcript, video_duration)
        face_analysis = _estimate_face(audio_analysis)
        print(f"[PIPELINE] Scores: conf={audio_analysis.confidence_score}, expr={face_analysis.avg_expression_score}, eye={face_analysis.avg_eye_contact_score}")

        # ── Select highlight ──
        highlight = select_highlight(audio_analysis, face_analysis, transcript, video_duration)

        # ── Resume + Clip in parallel ──
        t = time.time()
        JobQueue.update_job(job_id, JobStatus.GENERATING_RESUME, 55, "Building resume & clip...")

        def do_resume():
            rd = generate_resume(
                transcript=transcript, audio_analysis=audio_analysis,
                face_analysis=face_analysis, user_name=request.user_name,
                user_university=request.user_university, user_branch=request.user_branch)
            pdf = generate_pdf(
                resume=rd, audio=audio_analysis, face=face_analysis,
                user_name=request.user_name, user_university=request.user_university,
                user_branch=request.user_branch, job_id=job_id)
            return rd, pdf

        with ThreadPoolExecutor(max_workers=2) as ex:
            rf = ex.submit(do_resume)
            cf = ex.submit(generate_highlight_clip, video_path, job_id, highlight)
            resume_data, pdf_path = rf.result()
            clip_path = cf.result()
        log("Resume+Clip", t)

        # ── Upload in parallel ──
        t = time.time()
        JobQueue.update_job(job_id, JobStatus.UPLOADING_RESULTS, 80, "Uploading results...")
        with ThreadPoolExecutor(max_workers=2) as ex:
            pf = ex.submit(upload_pdf, pdf_path, request.user_id, job_id)
            uf = ex.submit(upload_video_clip, clip_path, request.user_id, job_id)
            resume_url = pf.result()
            clip_url = uf.result()
        log("Upload", t)

        # ── Save ──
        JobQueue.update_job(job_id, JobStatus.UPLOADING_RESULTS, 95, "Saving...")
        result = ProcessingResult(
            transcript=transcript.full_text,
            resume_pdf_url=resume_url,
            highlight_clip_url=clip_url,
            confidence_score=audio_analysis.confidence_score,
            energy_score=round(audio_analysis.average_energy * 1000, 1),
            expression_score=face_analysis.avg_expression_score,
            eye_contact_score=face_analysis.avg_eye_contact_score,
            skills_extracted=resume_data.skills,
            highlight_start=highlight.start_time,
            highlight_end=highlight.end_time,
        )
        update_submission_completed(request.submission_id, result)
        JobQueue.complete_job(job_id, result)

        total = time.time() - t0
        print(f"[PIPELINE] ✅ DONE in {total:.1f}s ({total/60:.1f} min)")

    except Exception as e:
        total = time.time() - t0
        print(f"[ERROR] Pipeline failed after {total:.1f}s: {e}")
        print(traceback.format_exc())
        JobQueue.fail_job(job_id, str(e))
        try:
            update_submission_failed(request.submission_id, str(e))
        except:
            pass
    finally:
        try:
            cleanup_temp_files(job_id)
        except:
            pass


def _score_from_transcript(transcript: TranscriptionResult, duration: float) -> AudioAnalysis:
    """
    Calculate scores from transcript text — NO librosa needed.
    Uses word count, speaking rate, vocabulary diversity as proxies.
    """
    text = transcript.full_text
    words = text.split()
    word_count = len(words)
    
    # Speaking rate (words per minute)
    wpm = (word_count / max(duration, 1)) * 60
    
    # Unique words ratio (vocabulary diversity)
    unique_ratio = len(set(w.lower() for w in words)) / max(word_count, 1)
    
    # Average word length (complexity)
    avg_word_len = sum(len(w) for w in words) / max(word_count, 1)
    
    # Calculate confidence score
    # Good: 120-180 WPM, diverse vocabulary, longer words
    wpm_score = max(0, min(100, 100 - abs(wpm - 150) * 0.8))
    vocab_score = min(100, unique_ratio * 200)
    length_score = min(100, avg_word_len * 20)
    word_count_score = min(100, word_count * 0.5)  # More words = more confident
    
    confidence = round((wpm_score * 0.3 + vocab_score * 0.25 + length_score * 0.15 + word_count_score * 0.3), 1)
    confidence = max(35, min(95, confidence))  # Clamp between 35-95
    
    # Estimate energy from WPM
    energy = max(0.01, min(0.1, wpm / 2000))
    
    return AudioAnalysis(
        confidence_score=confidence,
        average_energy=energy,
        speaking_rate_wpm=round(wpm, 1),
        pitch_mean=180.0,
        pitch_std=30.0,
        energy_std=0.02,
        silence_ratio=max(0.1, 1 - (word_count / max(duration * 2.5, 1))),
        speech_segments=[],
    )


def _estimate_face(audio: AudioAnalysis) -> FaceAnalysis:
    """Estimate face scores from voice analysis."""
    conf = audio.confidence_score
    return FaceAnalysis(
        face_detected=True,
        avg_eye_contact_score=round(max(30, min(95, conf * 0.85 + 8)), 1),
        avg_expression_score=round(max(30, min(95, conf * 0.9 + 5)), 1),
        per_second_scores=[],
    )
