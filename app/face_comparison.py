"""Face comparison: extract best frame from video and compare against INE photo.

Uses InsightFace (ArcFace + ONNX Runtime) — no TensorFlow/Keras dependency.

Design notes:
- Two-stage cascade. The buffalo_l detector + r50 recognition model embed every
  frame cheaply and rank them by similarity to the INE; the stronger glint360k
  ResNet100 model (antelopev2) then re-scores only the INE and the top-K frames.
  Validated on 64 real cases: the r100 comparator lifts the genuine score floor
  (min 35 -> 42) and recovers the false rejects that r50 produced on hard
  appearance-gap pairs, while the cascade keeps r100 cost to K+1 embeddings.
- Detection is shared across both stages (run once, batched recognition).
- Loads only detection + recognition (`allowed_modules`); the unused 2D/3D
  landmark and gender-age models are skipped.

NOTE: r100 scores run higher than r50, so the decision threshold must be
recalibrated for this scale (final FAR calibration needs impostor samples).
"""

import os
import glob
import logging
from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MAX_INE_DIMENSION = 1200
_DET_SIZE = (640, 640)
_CASCADE_K = 2  # frames re-scored by the strong model (validated: keeps all genuine above threshold)

# --- InsightFace models (lazy loaded) ---
_face_app = None
_strong_rec = None


def _get_face_app():
    """Lazy-load InsightFace with detection + recognition only.

    Same models and alignment as the full buffalo_l pipeline, so embeddings
    (and therefore the similarity score) are unchanged — just without the
    unused landmark and gender-age inference on every face.
    """
    global _face_app
    if _face_app is not None:
        return _face_app

    from insightface.app import FaceAnalysis

    _face_app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    _face_app.prepare(ctx_id=-1, det_size=_DET_SIZE)
    logger.info("InsightFace loaded (buffalo_l, detection+recognition, CPU)")
    return _face_app


def _get_strong_rec():
    """Lazy-load the glint360k ResNet100 recognition model (antelopev2).

    Downloads the antelopev2 pack on first use if it isn't cached. Returns None
    if it can't be loaded, so the caller can fall back to the r50 model.
    """
    global _strong_rec
    if _strong_rec is not None:
        return _strong_rec

    try:
        from insightface.model_zoo import get_model
        from insightface.utils import storage

        root = os.path.expanduser("~/.insightface/models")
        found = glob.glob(os.path.join(root, "**", "glintr100.onnx"), recursive=True)
        if not found:
            storage.ensure_available("models", "antelopev2", root="~/.insightface")
            found = glob.glob(os.path.join(root, "**", "glintr100.onnx"), recursive=True)
        if not found:
            raise FileNotFoundError("glintr100.onnx (antelopev2) not found")

        rec = get_model(found[0], providers=["CPUExecutionProvider"])
        rec.prepare(ctx_id=-1)
        logger.info("Strong recognition model loaded (glint360k r100, CPU)")
        _strong_rec = rec
    except Exception as exc:
        logger.warning("Strong recognition model unavailable, falling back to r50: %s", exc)
        _strong_rec = None
    return _strong_rec


def warmup_deepface():
    """Pre-load both recognition models so the first request is fast."""
    try:
        _get_face_app()
        _get_strong_rec()
        logger.info("InsightFace warm-up complete.")
    except Exception as exc:
        logger.warning("InsightFace warm-up failed: %s", exc)


# --- Data classes ---

@dataclass
class BestFrameInfo:
    frame_index: int
    total_frames: int
    quality_score: float
    sharpness: float
    lighting_score: float


@dataclass
class FaceMatchResult:
    similarity_score: float
    is_match: bool
    threshold_used: float
    best_frame_info: dict
    error: Optional[str] = None


# --- Helpers ---

def _laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _lighting_score(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    exposure = max(0.0, 1.0 - abs(mean_val - 128.0) / 128.0)
    contrast = min(std_val / 64.0, 1.0)
    return float(np.clip(0.7 * exposure + 0.3 * contrast, 0.0, 1.0))


def _preprocess_ine_image(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim > _MAX_INE_DIMENSION:
        scale = _MAX_INE_DIMENSION / max_dim
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image


def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    dot = np.dot(emb1, emb2)
    norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    if norm < 1e-8:
        return 0.0
    return float(max(0.0, dot / norm) * 100)


def _detect_and_align(images: list[np.ndarray]):
    """Detect the largest face in each image and return aligned 112x112 crops.

    Largest-face selection handles the INE's faint holographic ghost portrait:
    the main photo is the bigger detection. Returns (crops, positions) where
    positions[j] is the index in `images` that produced crops[j]; images with no
    detected face are omitted. The crops feed both recognition models, so
    detection and alignment run only once.
    """
    from insightface.utils import face_align

    det = _get_face_app().models["detection"]
    crops, positions = [], []
    for i, image in enumerate(images):
        bboxes, kpss = det.detect(image, max_num=0, metric="default")
        if bboxes.shape[0] == 0:
            continue
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        kps = kpss[int(np.argmax(areas))]
        crops.append(face_align.norm_crop(image, landmark=kps, image_size=112))
        positions.append(i)
    return crops, positions


def extract_frames_for_comparison(video_path: str, target_frames: int = 8) -> np.ndarray:
    """Extract frames at original resolution for face comparison.

    Only extracts 8 frames (not 16) since we only need a few good candidates.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 0
        while cap.grab():
            total += 1
        cap.release()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot reopen video: {video_path}")

    if total < 4:
        cap.release()
        raise ValueError(f"Video too short: {total} frames")

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
                sampled[idx] = frame
        idx += 1

    cap.release()

    frames = [sampled[i] for i in sorted(sampled.keys())]
    if not frames:
        raise ValueError(f"Could not read any frames from: {video_path}")

    return np.array(frames)


# --- Face comparison ---

def compare_video_vs_ine(
    frames_bgr: np.ndarray,
    ine_image: np.ndarray,
    threshold: float = 50.0,
) -> FaceMatchResult:
    """Compare video frames against an INE photo with an r50 -> r100 cascade.

    1. Detect and align the INE and every frame once.
    2. The fast r50 model embeds all crops and ranks the frames by similarity to
       the INE.
    3. The strong r100 model re-scores only the INE and the top-K frames; the
       match score is the max r100 similarity over those frames. Taking the max
       (rather than fusing) recovers the one good frame a genuine user may show
       only briefly. Falls back to r50-only max if the strong model is missing.
    """
    ine_image = _preprocess_ine_image(ine_image)

    crops, positions = _detect_and_align([ine_image, *frames_bgr])
    if 0 not in positions:
        return FaceMatchResult(
            similarity_score=0.0, is_match=False, threshold_used=threshold,
            best_frame_info={},
            error="No se detectó rostro en la imagen de INE",
        )
    pos_to_crop = {p: c for p, c in zip(positions, crops)}
    frame_positions = [p for p in positions if p != 0]
    if not frame_positions:
        return FaceMatchResult(
            similarity_score=0.0, is_match=False, threshold_used=threshold,
            best_frame_info={},
            error="No se detectó rostro en los frames del video",
        )

    # Stage 1: fast model ranks frames by similarity to the INE
    fast = _get_face_app().models["recognition"]
    fast_feats = {p: f for p, f in zip(positions, fast.get_feat(crops))}
    ranked = sorted(
        frame_positions,
        key=lambda p: _cosine_similarity(fast_feats[p], fast_feats[0]),
        reverse=True,
    )

    # Stage 2: strong model re-scores the INE + top-K frames (or fall back to r50)
    strong = _get_strong_rec()
    if strong is not None:
        top = ranked[:_CASCADE_K]
        feats = strong.get_feat([pos_to_crop[0]] + [pos_to_crop[p] for p in top])
        ine_emb, frame_embs = feats[0], dict(zip(top, feats[1:]))
        scored = top
    else:
        ine_emb, frame_embs = fast_feats[0], fast_feats
        scored = frame_positions

    best_similarity = -1.0
    best_pos = -1
    for p in scored:
        similarity = _cosine_similarity(frame_embs[p], ine_emb)
        if similarity > best_similarity:
            best_similarity = similarity
            best_pos = p

    best_idx = best_pos - 1  # positions are offset by the INE at index 0
    gray = cv2.cvtColor(frames_bgr[best_idx], cv2.COLOR_BGR2GRAY)
    info = BestFrameInfo(
        frame_index=best_idx,
        total_frames=len(frames_bgr),
        quality_score=round(best_similarity / 100, 4),
        sharpness=round(_laplacian_sharpness(gray), 2),
        lighting_score=round(_lighting_score(gray), 4),
    )

    logger.info(
        "Face match: best frame=%d/%d, similarity=%.2f%%, match=%s",
        best_idx, len(frames_bgr), best_similarity, best_similarity >= threshold,
    )

    return FaceMatchResult(
        similarity_score=round(best_similarity, 2),
        is_match=best_similarity >= threshold,
        threshold_used=threshold,
        best_frame_info=asdict(info),
    )
