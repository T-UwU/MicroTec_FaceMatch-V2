"""Video preprocessing: frame extraction + landmark detection."""

import cv2
import numpy as np
from pathlib import Path
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
import mediapipe as mp

_MODEL_PATH = str(Path(__file__).parent.parent / "model" / "face_landmarker.task")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def extract_frames(video_path: str, target_frames: int = 16, target_size: int = 224) -> np.ndarray:
    """Extract frames uniformly from video. Returns (T, H, W, 3) BGR uint8."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Count frames
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 0
        while cap.grab():
            total += 1
        cap.release()
        if total < 4:
            raise ValueError(f"Video too short: {total} frames")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot reopen video: {video_path}")

    if total < 4:
        cap.release()
        raise ValueError(f"Video too short: {total} frames")

    # Uniform sampling
    indices = set(np.linspace(0, total - 1, min(target_frames, total), dtype=int))
    sampled = {}
    idx = 0

    while True:
        ret = cap.grab()
        if not ret:
            break
        if idx in indices:
            ret2, frame = cap.retrieve()
            if ret2:
                sampled[idx] = cv2.resize(frame, (target_size, target_size))
        idx += 1

    cap.release()

    frames = [sampled[i] for i in sorted(sampled.keys())]
    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from: {video_path}")

    # Pad if needed
    while len(frames) < target_frames:
        frames.append(frames[-1].copy())

    return np.array(frames[:target_frames])


class MultipleFacesError(Exception):
    """Raised when more than one face is detected."""
    pass


def detect_landmarks(frames: np.ndarray) -> np.ndarray:
    """Detect 68 facial landmarks per frame. Returns (T, 68, 2) float32.
    
    Raises MultipleFacesError if more than one face is detected in any frame.
    """
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        num_faces=2,  # Detect up to 2 to check for multiple faces
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    landmarker = FaceLandmarker.create_from_options(options)
    landmarks_list = []
    multi_face_count = 0

    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(image)

        if result.face_landmarks:
            if len(result.face_landmarks) > 1:
                multi_face_count += 1
            lm = result.face_landmarks[0]
            landmarks_list.append(np.array([[p.x, p.y] for p in lm[:68]]))
        else:
            landmarks_list.append(np.zeros((68, 2)))

    landmarker.close()

    # If multiple faces detected in 3+ frames, reject
    if multi_face_count >= 3:
        raise MultipleFacesError(
            f"Se detectaron múltiples rostros en {multi_face_count} frames. "
            "Solo debe aparecer una persona en el video."
        )

    return np.array(landmarks_list, dtype=np.float32)


def preprocess_video(video_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Full preprocessing pipeline.
    
    Returns:
        frames_tensor: (1, T, 3, H, W) normalized for model
        landmarks_tensor: (1, T, 68, 2) for model
        metadata: dict with preprocessing info
    """
    frames = extract_frames(video_path)
    landmarks = detect_landmarks(frames)

    # Metadata for reasoning
    total_frames = len(frames)
    face_detected = int((landmarks.sum(axis=(1, 2)) > 0).sum())
    landmark_std = float(landmarks[landmarks.sum(axis=(1, 2)) > 0].std()) if face_detected > 0 else 0.0

    metadata = {
        "frames_analyzed": total_frames,
        "face_detected_frames": face_detected,
        "landmark_variability": round(landmark_std, 4),
    }

    # BGR -> RGB, normalize
    rgb = frames[..., ::-1].copy().astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD

    # (T, H, W, 3) -> (1, T, 3, H, W)
    frames_tensor = normalized.transpose(0, 3, 1, 2)[np.newaxis].astype(np.float32)
    landmarks_tensor = landmarks[np.newaxis].astype(np.float32)

    return frames_tensor, landmarks_tensor, metadata

