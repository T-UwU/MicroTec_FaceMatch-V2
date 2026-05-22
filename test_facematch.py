"""Standalone test for the face match module only.

Usage: python test_facematch.py <ine_image> <video> [threshold]
"""

import sys
import time
import json

import cv2

from app.face_comparison import (
    compare_video_vs_ine,
    extract_frames_for_comparison,
    warmup_deepface,
)


def main():
    ine_path = sys.argv[1]
    video_path = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0

    ine_bgr = cv2.imread(ine_path)
    if ine_bgr is None:
        raise SystemExit(f"Cannot read INE image: {ine_path}")

    print("Warming up models (downloads buffalo_l on first run)...")
    t0 = time.time()
    warmup_deepface()
    print(f"Warmup: {time.time() - t0:.2f}s\n")

    frames = extract_frames_for_comparison(video_path)
    print(f"Extracted {len(frames)} frames from video")

    t0 = time.time()
    result = compare_video_vs_ine(frames, ine_bgr, threshold=threshold)
    elapsed = (time.time() - t0) * 1000

    print(json.dumps({
        "match": result.is_match,
        "similarity_score": result.similarity_score,
        "threshold_used": result.threshold_used,
        "best_frame_info": result.best_frame_info,
        "error": result.error,
    }, indent=2, ensure_ascii=False))
    print(f"\nface_match time: {elapsed:.0f} ms")


if __name__ == "__main__":
    main()
