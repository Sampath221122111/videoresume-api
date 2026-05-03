"""
Pipeline — ULTRA FAST.
Key: Transcribe + Tone + Face ALL run in parallel (3 threads).
Then Resume + Clip in parallel. Then uploads in parallel.
Target: < 60 seconds for a 1-minute video.
"""

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from app.models.schemas import JobStatus, ProcessingResult, ProcessVideoRequest
from app.utils.queue import JobQueue
from app.services import (
    download_video, validate_video, extract_audio, get_video_duration,
    transcribe_audio, analyze_tone, analyze_face,
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
        elapsed = time.time() - start
        print(f"[PIPELINE] {stage}: {elapsed:.1f}s")

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

        # ── PARALLEL: Transcribe + Tone + Face (3 threads!) ──
        t = time.time()
        JobQueue.update_job(job_id, JobStatus.TRANSCRIBING, 25, "Analyzing everything...")
        with ThreadPoolExecutor(max_workers=3) as ex:
            t_future = ex.submit(transcribe_audio, audio_path)
            tone_future = ex.submit(_safe_tone, audio_path)
            face_future = ex.submit(analyze_face, video_path)

            transcript = t_future.result()
            audio_analysis_partial = tone_future.result()
            face_analysis = face_future.result()
        log("Transcribe+Tone+Face (3-way parallel)", t)

        if not transcript.full_text.strip():
            raise ValueError("No speech detected. Record with clear audio.")

        # Finalize tone analysis with transcript data
        audio_analysis = _merge_tone(audio_analysis_partial, transcript)

        highlight = select_highlight(audio_analysis, face_analysis, transcript, video_duration)

        # ── PARALLEL: Resume/PDF + Clip ──
        t = time.time()
        JobQueue.update_job(job_id, JobStatus.GENERATING_RESUME, 55, "Building resume & clip...")

        def do_resume():
            rd = generate_resume(transcript=transcript, audio_analysis=audio_analysis,
                face_analysis=face_analysis, user_name=request.user_name,
                user_university=request.user_university, user_branch=request.user_branch)
            pdf = generate_pdf(resume=rd, audio=audio_analysis, face=face_analysis,
                user_name=request.user_name, user_university=request.user_university,
                user_branch=request.user_branch, job_id=job_id)
            return rd, pdf

        with ThreadPoolExecutor(max_workers=2) as ex:
            rf = ex.submit(do_resume)
            cf = ex.submit(generate_highlight_clip, video_path, job_id, highlight)
            resume_data, pdf_path = rf.result()
            clip_path = cf.result()
        log("Resume+Clip (parallel)", t)

        # ── PARALLEL: Upload PDF + Clip ──
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
            transcript=transcript.full_text, resume_pdf_url=resume_url, highlight_clip_url=clip_url,
            confidence_score=audio_analysis.confidence_score,
            energy_score=round(audio_analysis.average_energy * 1000, 1),
            expression_score=face_analysis.avg_expression_score,
            eye_contact_score=face_analysis.avg_eye_contact_score,
            skills_extracted=resume_data.skills,
            highlight_start=highlight.start_time, highlight_end=highlight.end_time,
        )
        update_submission_completed(request.submission_id, result)
        JobQueue.complete_job(job_id, result)

        total = time.time() - t0
        print(f"[PIPELINE] ✅ DONE in {total:.1f}s ({total/60:.1f} min)")

    except Exception as e:
        print(f"[ERROR] Pipeline failed after {time.time()-t0:.1f}s: {e}")
        print(traceback.format_exc())
        JobQueue.fail_job(job_id, str(e))
        try: update_submission_failed(request.submission_id, str(e))
        except: pass
    finally:
        try: cleanup_temp_files(job_id)
        except: pass


def _safe_tone(audio_path):
    """Run tone analysis without transcript (partial)."""
    try:
        from app.services.tone_service import analyze_tone_only
        return analyze_tone_only(audio_path)
    except (ImportError, AttributeError):
        # Fallback: run with empty transcript
        from app.models.schemas import TranscriptionResult
        empty = TranscriptionResult(full_text="", segments=[], language="en")
        from app.services import analyze_tone
        return analyze_tone(audio_path, empty)


def _merge_tone(audio_analysis, transcript):
    """Merge transcript word count into tone analysis for better confidence."""
    word_count = len(transcript.full_text.split())
    # Boost confidence if they spoke enough words
    if word_count > 50 and audio_analysis.confidence_score < 80:
        audio_analysis.confidence_score = min(100, audio_analysis.confidence_score + 5)
    return audio_analysis
