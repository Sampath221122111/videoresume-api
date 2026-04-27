"""
Face Analysis Service — uses MediaPipe (new API, no mp.solutions).
Analyzes face presence, eye contact, and expression.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from app.models.schemas import FaceAnalysis
import os
import urllib.request
import tempfile


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(tempfile.gettempdir(), "face_landmarker.task")


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[FACE] Downloading face landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[FACE] Model downloaded.")


def analyze_face(video_path: str) -> FaceAnalysis:
    try:
        ensure_model()
    except Exception as e:
        print(f"[FACE] Model download failed: {e}, using fallback")
        return analyze_face_opencv(video_path)

    try:
        return analyze_face_mediapipe(video_path)
    except Exception as e:
        print(f"[FACE] MediaPipe failed: {e}, using OpenCV fallback")
        return analyze_face_opencv(video_path)


def analyze_face_mediapipe(video_path: str) -> FaceAnalysis:
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return FaceAnalysis(face_detected=False)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, int(fps / 2) * 4)  # Analyze ~2 frames/sec instead of 15 for speed

    detection_scores = []
    expression_scores = []
    eye_contact_scores = []
    per_second_expr = []
    sec_buf = []
    cur_sec = 0

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_time = frame_idx / fps

            if frame_idx % sample_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(frame_time * 1000)

                try:
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                except Exception:
                    frame_idx += 1
                    continue

                if result.face_landmarks and len(result.face_landmarks) > 0:
                    landmarks = result.face_landmarks[0]
                    detection_scores.append(1.0)

                    nose = landmarks[1]
                    left_eye = landmarks[33]
                    right_eye = landmarks[263]

                    eye_cx = (left_eye.x + right_eye.x) / 2
                    eye_cy = (left_eye.y + right_eye.y) / 2
                    offset_x = abs(nose.x - eye_cx)
                    offset_y = abs(nose.y - eye_cy)
                    eye_score = max(0, 100 - (offset_x * 300 + offset_y * 200))
                    eye_contact_scores.append(eye_score)

                    expr_score = 50.0
                    if result.face_blendshapes and len(result.face_blendshapes) > 0:
                        blendshapes = result.face_blendshapes[0]
                        smile_score = 0
                        for bs in blendshapes:
                            if "smile" in bs.category_name.lower() or "mouth_smile" in bs.category_name.lower():
                                smile_score = max(smile_score, bs.score)
                            if "jaw_open" in bs.category_name.lower():
                                smile_score = max(smile_score, bs.score * 0.5)
                        expr_score = min(100, smile_score * 150 + 30)

                    expression_scores.append(expr_score)
                    sec_buf.append(expr_score)
                else:
                    detection_scores.append(0.0)
                    eye_contact_scores.append(0.0)
                    expression_scores.append(0.0)
                    sec_buf.append(0.0)

            if frame_time >= cur_sec + 1:
                if sec_buf:
                    per_second_expr.append(float(np.mean(sec_buf)))
                sec_buf = []
                cur_sec = int(frame_time)

            frame_idx += 1

    cap.release()

    if sec_buf:
        per_second_expr.append(float(np.mean(sec_buf)))

    face_detected = len(detection_scores) > 0 and np.mean(detection_scores) > 0.3

    return FaceAnalysis(
        face_detected=face_detected,
        face_detection_confidence=round(float(np.mean(detection_scores)) * 100, 1) if detection_scores else 0,
        avg_eye_contact_score=round(float(np.mean(eye_contact_scores)), 1) if eye_contact_scores else 0,
        avg_expression_score=round(float(np.mean(expression_scores)), 1) if expression_scores else 0,
        expression_timeline=[round(e, 1) for e in per_second_expr],
    )


def analyze_face_opencv(video_path: str) -> FaceAnalysis:
    """Fallback: use OpenCV Haar cascade for basic face detection."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return FaceAnalysis(face_detected=False)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, int(fps / 2) * 4)  # Analyze ~2 frames/sec instead of 15 for speed

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    detections = []
    per_second = []
    sec_buf = []
    cur_sec = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_time = frame_idx / fps

        if frame_idx % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            detected = len(faces) > 0
            detections.append(1.0 if detected else 0.0)
            score = 60.0 if detected else 0.0
            sec_buf.append(score)

        if frame_time >= cur_sec + 1:
            if sec_buf:
                per_second.append(float(np.mean(sec_buf)))
            sec_buf = []
            cur_sec = int(frame_time)

        frame_idx += 1

    cap.release()
    if sec_buf:
        per_second.append(float(np.mean(sec_buf)))

    face_detected = len(detections) > 0 and np.mean(detections) > 0.3

    return FaceAnalysis(
        face_detected=face_detected,
        face_detection_confidence=round(float(np.mean(detections)) * 100, 1) if detections else 0,
        avg_eye_contact_score=50.0 if face_detected else 0,
        avg_expression_score=50.0 if face_detected else 0,
        expression_timeline=[round(e, 1) for e in per_second],
    )
