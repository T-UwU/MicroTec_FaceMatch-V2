"""Face Anti-Spoofing API.

Single endpoint handles both liveness-only and liveness + face match:
- POST /v1/liveness (video only) → liveness check
- POST /v1/liveness (video + ine_image) → liveness check, then face match if REAL
"""

import uuid
import time
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles

from .model import AntiSpoofingModel
from .preprocessing import preprocess_video, MultipleFacesError
from .reasoning import generate_reasoning
from .face_comparison import (
    compare_video_vs_ine, warmup_deepface, extract_frames_for_comparison,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

model: AntiSpoofingModel = None
_executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = AntiSpoofingModel()
    logger.info(f"Model loaded: {model.version}")
    warmup_deepface()
    yield
    _executor.shutdown(wait=False)


app = FastAPI(title="Face Anti-Spoofing API", version="2.0.0", lifespan=lifespan)

TEMP_DIR = Path("/tmp/antispoofing")
TEMP_DIR.mkdir(exist_ok=True)
MAX_SIZE_MB = 50
ALLOWED_VIDEO_EXT = {".mp4", ".webm", ".avi", ".mov"}
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _process_liveness(video_path: str) -> tuple:
    """Run liveness pipeline (CPU-bound, runs in thread pool)."""
    frames, landmarks, metadata = preprocess_video(video_path)
    result = model.predict(frames, landmarks)
    return result, metadata


def _process_face_match(video_path: str, ine_bgr: np.ndarray, threshold: float):
    """Run face match pipeline (CPU-bound, runs in thread pool)."""
    frames_bgr = extract_frames_for_comparison(video_path)
    return compare_video_vs_ine(frames_bgr, ine_bgr, threshold=threshold)


@app.post("/v1/liveness")
async def verify_liveness(
    video: UploadFile = File(...),
    ine_image: Optional[UploadFile] = File(None),
    match_threshold: Optional[float] = Form(50.0),
):
    """Verify liveness, optionally compare against INE photo."""
    start = time.time()
    request_id = str(uuid.uuid4())
    temp_video = None

    try:
        # --- Validate video ---
        ext = Path(video.filename or "").suffix.lower()
        if ext not in ALLOWED_VIDEO_EXT:
            raise HTTPException(400, f"Video format {ext} not allowed. Use: {ALLOWED_VIDEO_EXT}")

        video_content = await video.read()
        if len(video_content) > MAX_SIZE_MB * 1024 * 1024:
            raise HTTPException(400, f"Video exceeds {MAX_SIZE_MB}MB limit")

        temp_video = TEMP_DIR / f"{request_id}{ext}"
        with open(temp_video, "wb") as f:
            f.write(video_content)

        # --- Validate INE if provided ---
        ine_bgr = None
        if ine_image is not None:
            ine_ext = Path(ine_image.filename or "").suffix.lower()
            if ine_ext not in ALLOWED_IMAGE_EXT:
                raise HTTPException(400, f"Image format {ine_ext} not allowed. Use: {ALLOWED_IMAGE_EXT}")

            ine_content = await ine_image.read()
            if len(ine_content) > MAX_SIZE_MB * 1024 * 1024:
                raise HTTPException(400, f"INE image exceeds {MAX_SIZE_MB}MB limit")

            ine_arr = np.frombuffer(ine_content, dtype=np.uint8)
            ine_bgr = cv2.imdecode(ine_arr, cv2.IMREAD_COLOR)
            if ine_bgr is None:
                raise HTTPException(400, "Cannot decode INE image")

        # --- Step 1: Liveness (in thread pool) ---
        loop = asyncio.get_event_loop()
        result, metadata = await loop.run_in_executor(
            _executor, _process_liveness, str(temp_video)
        )

        score = result["score"]
        decision = "SPOOF" if score > model.threshold else "REAL"
        reasoning = generate_reasoning(score, metadata)

        logger.info(f"[{request_id}] Liveness: {decision} score={score:.4f}")

        response = {
            "request_id": request_id,
            "model_version": model.version,
            "decision": decision,
            "confidence": {
                "real": round(1 - score, 4),
                "spoof": round(score, 4),
            },
            "reasoning": reasoning,
        }

        # --- Step 2: Face match (in thread pool, only if REAL + INE) ---
        if ine_bgr is not None:
            if decision == "REAL":
                match_result = await loop.run_in_executor(
                    _executor, _process_face_match,
                    str(temp_video), ine_bgr, match_threshold,
                )

                response["face_match"] = {
                    "match": match_result.is_match,
                    "similarity_score": match_result.similarity_score,
                    "threshold_used": match_result.threshold_used,
                    "best_frame_info": match_result.best_frame_info,
                }
                if match_result.error:
                    response["face_match"]["error"] = match_result.error

                logger.info(
                    f"[{request_id}] Match: {match_result.is_match} "
                    f"similarity={match_result.similarity_score:.2f}%"
                )
            else:
                response["face_match"] = {
                    "skipped": True,
                    "reason": "Liveness check failed (SPOOF detected)",
                }

        elapsed_ms = (time.time() - start) * 1000
        response["processing_time_ms"] = round(elapsed_ms)
        return response

    except HTTPException:
        raise
    except MultipleFacesError as e:
        elapsed_ms = (time.time() - start) * 1000
        return {
            "request_id": request_id,
            "model_version": model.version,
            "decision": "REJECTED",
            "rejection_reason": "multiple_faces",
            "message": str(e),
            "processing_time_ms": round(elapsed_ms),
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}", exc_info=True)
        raise HTTPException(500, "Internal error processing video")
    finally:
        if temp_video and temp_video.exists():
            temp_video.unlink()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_version": model.version if model else None,
        "model_loaded": model is not None,
    }


# --- Demo web  --- 
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
