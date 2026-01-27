import os
import re
import cv2
import requests
import numpy as np
from typing import Tuple

# Ensure the image is grayscale; converts BGR to GRAY if needed
def ensure_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

# Estimate near-horizontal skew and rotate images to correct it
def deskew_soft(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(g, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
    angle = 0.0
    if lines is not None and len(lines) > 0:
        angles = []
        for rho_theta in lines[:50]:
            _, th = rho_theta[0] 
            a = (th - np.pi / 2) * 180.0 / np.pi
            if -15 <= a <= 15:
                angles.append(a)
        if angles:
            angle = float(np.median(angles))
    
    # Rotate the image
    M = cv2.getRotationMatrix2D((gray.shape[1] / 2, gray.shape[0] / 2), angle, 1.0)
    rot = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rot, angle

# Download and decode an image from a local path or URL with size guards
def load_image_local_or_url(source: str) -> np.ndarray:
    source = source.strip()
    
    # 1. Handle Local File
    if os.path.isfile(source):
        print(f"Detected local file: {source}")
        try:
            with open(source, "rb") as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("File exists but could not be decoded as an image.")
                return img
        except Exception as e:
            raise ValueError(f"Failed to read local file: {e}")

    # 2. Handle URL
    url = source
    if "drive.google.com" in url:
        match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
        if match:
            file_id = match.group(1)
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            print(f"Detected Google Drive link. Converted to: {url}")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    with requests.get(url, timeout=30, stream=True, headers=headers) as r:
        r.raise_for_status()
        max_bytes = 20 * 1024 * 1024  # 20MB limit
        data = bytearray()
        for chunk in r.iter_content(64 * 1024):
            if chunk:
                data.extend(chunk)
                if len(data) > max_bytes:
                    raise ValueError("Image too large (>20MB)")
                    
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image from URL")
        
    # 3. Resize if too massive (safety guard)
    h, w = img.shape[:2]
    max_dim = 6000
    if max(h, w) > max_dim:
        s = float(max_dim) / float(max(h, w))
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        
    return img

# Normalize a glyph image (digits and numbers) by cropping, denoise, resize
def _prep_for_match(img: np.ndarray, target_h: int = 64, target_w: int = 48) -> np.ndarray:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        g = g[max(0, y - 2): y + h + 2, max(0, x - 2): x + w + 2]
    g = cv2.resize(g, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    return g

# Produce a binary image tuned for line/row detection
def binarize_for_lines(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    return th

# Normalize a row image to a fixed height while preserving aspect
def normalize_row_height(row_img: np.ndarray, target_h: int = 32) -> np.ndarray:
    h, w = row_img.shape[:2]
    if h == 0 or w == 0:
        return row_img
    scale = target_h / h
    nw = max(1, int(round(w * scale)))
    return cv2.resize(row_img, (nw, target_h), interpolation=cv2.INTER_LINEAR)

# Crop the image between y0..y1 with a minimal vertical pad and return the crop
def clip_with_min_pad(img: np.ndarray, y0: int, y1: int, pad: int = 2) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, _ = img.shape[:2]
    y0 = max(0, y0 - pad)
    y1 = min(h, y1 + pad)
    return img[y0:y1, :], (y0, y1)