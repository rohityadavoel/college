"""
Unified fast validator for image integrity and quality.
Optimized for low-latency inference.
"""
import cv2
import numpy as np
from typing import Dict

# Thresholds
STRICT_REJECT_THRESHOLD = 90.0
CLEAR_IMAGE_THRESHOLD = 110.0

def validate_image(path: str) -> Dict:
    """
    Single-pass validation:
    1. Read bytes
    2. Check magic headers
    3. Decode with OpenCV
    4. Calculate Laplacian Variance (Downsampled)
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception as e:
        return {"overall_ok": False, "error": f"Read error: {e}"}

    if not data:
        return {"overall_ok": False, "error": "File is empty"}

    # Magic Header Check
    h = data[:12]
    is_valid_header = (
        h.startswith(b"\xff\xd8") or           # JPEG
        h.startswith(b"\x89PNG\r\n\x1a\n") or  # PNG
        h[:4] == b"RIFF" and h[8:12] == b"WEBP" # WEBP
    )
    if not is_valid_header:
        return {"overall_ok": False, "error": "Invalid image format (must be JPEG, PNG, or WEBP)"}

    # Decode
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"overall_ok": False, "error": "Image corruption (OpenCV decode failed)"}

    # Quality check (Blur)
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Downsample if very large to speed up Laplacian (1024px is plenty)
    h, w = gray.shape
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        gray = cv2.resize(gray, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 3. Laplacian Variance
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if variance < STRICT_REJECT_THRESHOLD:
        return {
            "overall_ok": False, 
            "error": f"Image too blurry (var={variance:.1f})",
            "laplacian_variance": variance
        }

    return {
        "overall_ok": True,
        "laplacian_variance": variance,
        "quality_status": "clear" if variance > CLEAR_IMAGE_THRESHOLD else "soft",
        "dimensions": f"{w}x{h}"
    }
