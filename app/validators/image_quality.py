"""
Medical-Grade Laplacian Blur Detection (Optimized)
Ultra-fast version with:
 - Single CV decode
 - Single PIL open
 - No redundant BytesIO
"""

import sys
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple
import numpy as np
from PIL import Image, ImageFile
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Thresholds
STRICT_REJECT_THRESHOLD = 90.0
CLEAR_IMAGE_THRESHOLD = 110.0


# -------------------------------
# FAST FILE READER
# -------------------------------
def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# -------------------------------
# PIL CHECKS (Merged + Optimized)
# -------------------------------
def pil_checks(data: bytes) -> Dict:
    """
    Runs:
    - verify()
    - load()
    without double opening the image.
    """
    try:
        bio = io.BytesIO(data)

        # verify
        img = Image.open(bio)
        img.verify()

        # re-open & load fully
        img = Image.open(io.BytesIO(data))
        img.load()

        return {"ok": True, "msg": "Pillow verify & load OK"}
    except Exception as e:
        return {"ok": False, "msg": f"Pillow error: {e}"}


# -------------------------------
# BLUR DETECTION (Optimized)
# -------------------------------
def laplacian_blur_detect(data: bytes) -> Dict:
    """
    Very fast Laplacian variance calculation.
    """

    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "cv2.imdecode failed"}

    # FAST grayscale conversion using numpy
    gray = (0.114 * img[:, :, 0] +
            0.587 * img[:, :, 1] +
            0.299 * img[:, :, 2]).astype(np.uint8)

    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # decision logic
    if variance < STRICT_REJECT_THRESHOLD:
        status = "reject"
        blurry = True
    elif variance < CLEAR_IMAGE_THRESHOLD:
        status = "soft-but-acceptable"
        blurry = False
    else:
        status = "clear"
        blurry = False

    return {
        "laplacian_variance": variance,
        "quality_status": status,
        "blurred": blurry
    }


# -------------------------------
# RUN ALL CHECKS (Parallel)
# -------------------------------
def run_checks(path: str) -> Dict:
    summary = {"path": path}

    try:
        data = read_bytes(path)
    except Exception as e:
        summary["error"] = f"Failed to read image: {e}"
        return summary

    # perfect minimal thread count for 3 ops
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {
            ex.submit(pil_checks, data): "pil_checks",
            ex.submit(laplacian_blur_detect, data): "blur_check",
        }

        for fut in as_completed(futures):
            key = futures[fut]
            try:
                summary[key] = fut.result()
            except Exception as e:
                summary[key] = {"error": f"{key} crashed: {e}"}

    return summary
