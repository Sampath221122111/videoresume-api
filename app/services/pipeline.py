"""
Pipeline Orchestrator — OPTIMIZED for speed.
Runs tone+face analysis in PARALLEL, resume+clip in PARALLEL, uploads in PARALLEL.
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
        print(f"[PIPELINE] {stage}: {time.time()-start:.1f}s")

    try:
        t = time.time()
        JobQueue.update_job(job_id, JobStatus.DOWNLOADING, 5, "Downloading video...")
        update_submission_processing(request.submission_id)
        video_path = await download_video(request.video_url, job_id)
        log("Download", t)

        t = time.time()
        validation = validate_video(video_path)
        if not validation["is_valid"]:
            raise ValueError(f"Video validation failed: {validation['error']}")
        video_duration = validation["duration"]
        log("Validate", t)

        t = time.time()
        JobQueue.update_job(job_id, JobStatus.EXTRACTING_AUDIO, 15, "Extracting audio...")
        audio_path = extract_audio(video_path, job_id)
        log("Extract Audio", t)

        t = time.time()
        JobQueue.update_job(job_id, JobStatus.TRANSCRIBING, 25, "Transcribing speech...")
        transcript = transcribe_audio(audio_path)
        log("Transcribe", t)

        if not transcript.full_text.strip():
            raise ValueError("No speech detected. Record with clear audio.")

        # ── PARALLEL: Tone + Face ──
        t = time.time()
        JobQueue.update_job(job_id, JobStatus.ANALYZING_TONE, 35, "Analyzing voice & face...")
        with ThreadPoolExecutor(max_workers=2) as ex:
            tf = ex.submit(analyze_tone, audio_path, transcript)
            ff = ex.submit(analyze_face, video_path)
            audio_analysis = tf.result()
            face_analysis = ff.result()
        log("Tone+Face (parallel)", t)

        highlight = select_highlight(audio_analysis, face_analysis, transcript, video_duration)

        # ── PARALLEL: Resume/PDF + Clip ──
        t = time.time()
        JobQueue.update_job(job_id, JobStatus.GENERATING_RESUME, 60, "Building resume & clip...")

        def do_resume():
            rd = generate_resume(transcript=transcript, audio_analysis=audio_analysis, face_analysis=face_analysis, user_name=request.user_name, user_university=request.user_university, user_branch=request.user_branch)
            pdf = generate_pdf(resume=rd, audio=audio_analysis, face=face_analysis, user_name=request.user_name, user_university=request.user_university, user_branch=request.user_branch, job_id=job_id)
            return rd, pdf

        with ThreadPoolExecutor(max_workers=2) as ex:
            rf = ex.submit(do_resume)
            cf = ex.submit(generate_highlight_clip, video_path, job_id, highlight)
            resume_data, pdf_path = rf.result()
            clip_path = cf.result()
        log("Resume+Clip (parallel)", t)

        # ── PARALLEL: Upload PDF + Clip ──
        t = time.time()
        JobQueue.update_job(job_id, JobStatus.UPLOADING_RESULTS, 85, "Uploading results...")
        with ThreadPoolExecutor(max_workers=2) as ex:
            pf = ex.submit(upload_pdf, pdf_path, request.user_id, job_id)
            uf = ex.submit(upload_video_clip, clip_path, request.user_id, job_id)
            resume_url = pf.result()
            clip_url = uf.result()
        log("Upload (parallel)", t)

        JobQueue.update_job(job_id, JobStatus.UPLOADING_RESULTS, 95, "Saving...")
        result = ProcessingResult(
            transcript=transcript.full_text, resume_pdf_url=resume_url, highlight_clip_url=clip_url,
            confidence_score=audio_analysis.confidence_score, energy_score=round(audio_analysis.average_energy*1000,1),
            expression_score=face_analysis.avg_expression_score, eye_contact_score=face_analysis.avg_eye_contact_score,
            skills_extracted=resume_data.skills, highlight_start=highlight.start_time, highlight_end=highlight.end_time,
        )
        update_submission_completed(request.submission_id, result)
        JobQueue.complete_job(job_id, result)
        print(f"[PIPELINE] DONE in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")

    except Exception as e:
        print(f"[ERROR] Pipeline failed after {time.time()-t0:.1f}s: {e}")
        print(traceback.format_exc())
        JobQueue.fail_job(job_id, str(e))
        try: update_submission_failed(request.submission_id, str(e))
        except: pass
    finally:
        try: cleanup_temp_files(job_id)
        except: pass
