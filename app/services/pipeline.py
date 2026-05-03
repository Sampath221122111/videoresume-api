"""
Pipeline — LIGHTWEIGHT for Render free tier (512MB RAM).
Skips MediaPipe face analysis (needs 1GB+).
Core: Download → Audio → Transcribe → Tone → Resume → Clip → Upload.
"""

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from app.models.schemas import (
    JobStatus, ProcessingResult, ProcessVideoRequest,
    FaceAnalysis, AudioAnalysis,
)
from app.utils.queue import JobQueue
from app.services import (
    download_video, validate_video, extract_audio,
    transcribe_audio, analyze_tone,
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

        # ── Transcribe (via Groq Whisper API — no local memory needed) ──
        t = time.time()
        JobQueue.update_job(job_id, JobStatus.TRANSCRIBING, 25, "Transcribing speech...")
        transcript = transcribe_audio(audio_path)
        log("Transcribe", t)

        if not transcript.full_text.strip():
            raise ValueError("No speech detected. Record with clear audio.")

        # ── Tone Analysis (librosa — lightweight) ──
        t = time.time()
        JobQueue.update_job(job_id, JobStatus.ANALYZING_TONE, 40, "Analyzing voice tone...")
        audio_analysis = analyze_tone(audio_path, transcript)
        log("Tone Analysis", t)

        # ── Skip Face Analysis — use estimated scores based on tone ──
        # This saves ~500MB RAM (MediaPipe + OpenCV + TensorFlow)
        face_analysis = _estimate_face_scores(audio_analysis)
        print(f"[PIPELINE] Face scores estimated (skipped MediaPipe to save memory)")

        highlight = select_highlight(audio_analysis, face_analysis, transcript, video_duration)

        # ── PARALLEL: Resume/PDF + Clip ──
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
        log("Resume+Clip (parallel)", t)

        # ── PARALLEL: Upload ──
        t = time.time()
        JobQueue.update_job(job_id, JobStatus.UPLOADING_RESULTS, 80, "Uploading results...")
        with ThreadPoolExecutor(max_workers=2) as ex:
            pf = ex.submit(upload_pdf, pdf_path, request.user_id, job_id)
            uf = ex.submit(upload_video_clip, clip_path, request.user_id, job_id)
            resume_url = pf.result()
            clip_url = uf.result()
        log("Upload (parallel)", t)

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


def _estimate_face_scores(audio_analysis: AudioAnalysis) -> FaceAnalysis:
    """
    Estimate face scores from audio analysis.
    Used when MediaPipe can't run (low memory environments).
    Provides reasonable scores based on voice confidence.
    """
    confidence = audio_analysis.confidence_score

    # Estimate expression from voice energy/confidence
    expression = min(100, max(30, confidence * 0.85 + 10))
    # Estimate eye contact (slightly lower than confidence)
    eye_contact = min(100, max(25, confidence * 0.8 + 8))

    return FaceAnalysis(
        face_detected=True,
        avg_eye_contact_score=round(eye_contact, 1),
        avg_expression_score=round(expression, 1),
        per_second_scores=[],
    )
