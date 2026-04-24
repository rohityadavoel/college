"""
Ultra-Simple Multiprocessing Image Corruption Checker
Uses PIL + optional OpenCV.
"""

import sys
import io
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Dict

from PIL import Image

# Optional OpenCV
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except:
    OPENCV_AVAILABLE = False

MAX_DIMENSION = 20000


# ---------------------------------------------------------------------
# Basic single-file reader
# ---------------------------------------------------------------------
def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------
# Simple checks
# ---------------------------------------------------------------------
def check_non_empty(data: bytes) -> Tuple[str, bool, str]:
    if not data:
        return ("non_empty", False, "File empty")
    return ("non_empty", True, f"{len(data)} bytes")


def check_magic_header(data: bytes) -> Tuple[str, bool, str]:
    h = data[:12]
    if h.startswith(b"\xff\xd8"):
        return ("magic_header", True, "JPEG")
    if h.startswith(b"\x89PNG\r\n\x1a\n"):
        return ("magic_header", True, "PNG")
    if h[:6] in (b"GIF87a", b"GIF89a"):
        return ("magic_header", True, "GIF")
    if h[:4] == b"RIFF" and h[8:12] == b"WEBP":
        return ("magic_header", True, "WEBP")
    return ("magic_header", False, f"Unknown header {h[:8]!r}")


def check_pillow_load(data: bytes) -> Tuple[str, bool, str]:
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            w, h = img.size
        return ("pil_load", True, f"Loaded {w}x{h}")
    except Exception as e:
        return ("pil_load", False, f"PIL error: {e}")


def check_opencv(data: bytes) -> Tuple[str, bool, str]:
    if not OPENCV_AVAILABLE:
        return ("opencv", True, "OpenCV not installed")
    try:
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return ("opencv", False, "Decode failed")
        h, w = img.shape[:2]
        return ("opencv", True, f"OpenCV OK {w}x{h}")
    except Exception as e:
        return ("opencv", False, f"cv2 error: {e}")


# List of checks to run
CHECKS = [check_non_empty, check_magic_header, check_pillow_load, check_opencv]


# ---------------------------------------------------------------------
# Run all checks for a single file (runs inside processes)
# ---------------------------------------------------------------------
def inspect_file(path: str) -> Dict:
    result = {"path": path, "results": []}

    try:
        data = read_bytes(path)
    except Exception as e:
        result["error"] = f"Read error: {e}"
        return result

    # Run simple loop, no threads inside
    for check in CHECKS:
        cname, ok, msg = check(data)
        result["results"].append({
            "check": cname,
            "passed": ok,
            "message": msg
        })

    result["overall_ok"] = all(r["passed"] for r in result["results"])
    return result


# ---------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------
def print_report(r: Dict):
    print("=" * 55)
    print("File:", r["path"])

    if "error" in r:
        print("ERROR:", r["error"])
        print("=" * 55)
        return

    print("Overall:", "OK" if r["overall_ok"] else "BAD")
    for item in r["results"]:
        st = "PASS" if item["passed"] else "FAIL"
        print(f" - {item['check']:10} {st}   {item['message']}")
    print("=" * 55)


# ---------------------------------------------------------------------
# MAIN - MULTIPROCESSING
# ---------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python checker.py <img1> [img2...]")
        sys.exit(1)

    paths = sys.argv[1:]

    # Each image handled by a separate process
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as exe:
        futures = {exe.submit(inspect_file, p): p for p in paths}

        for fut in as_completed(futures):
            print_report(fut.result())
