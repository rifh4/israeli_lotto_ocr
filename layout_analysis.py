import math
import re
import cv2
import numpy as np
import pytesseract
# Added 'List' back here to fix the red error
from typing import List, Tuple, Optional

# Remove 'import config' (it is unused here, satisfying the warning)
from image_utils import binarize_for_lines

# Find plausible digit bounding boxes within a left-side number band
def cc_digit_boxes(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    H, W = gray.shape[:2]
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), 1)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h < int(0.40 * H) or h > int(0.96 * H):
            continue
        if w < max(3, int(0.012 * W)) or w > int(0.24 * W):
            continue
        roi = gray[y:y + h, x:x + w]
        if np.mean(roi) > 235:
            continue
        out.append((x, y, w, h))
    out.sort(key=lambda r: r[0])
    return out

# Merge adjacent digit boxes into token spans based on gap statistics
def merge_boxes_to_tokens(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda r: r[0])
    if len(boxes) == 1:
        x, y, w, h = boxes[0]
        return [(x, x + w)]
    gaps = [boxes[i + 1][0] - (boxes[i][0] + boxes[i][2]) for i in range(len(boxes) - 1)]
    pos = [g for g in gaps if g > 0]
    med = float(np.median(pos)) if pos else 0.0
    span_w = (boxes[-1][0] + boxes[-1][2]) - boxes[0][0]
    p25 = float(np.percentile(pos, 25)) if len(pos) >= 4 else med
    p75 = float(np.percentile(pos, 75)) if len(pos) >= 4 else med
    iqr = max(0.0, p75 - p25)
    base = 1.35 * med if med > 0 else 8.0
    if iqr > 0:
        base = min(base, p75 + 1.0 * iqr)
    thr = min(base, 0.05 * float(max(1, span_w)), 22.0)
    thr = max(6.0, thr)
    tokens: List[Tuple[int, int]] = []
    cur_x0 = boxes[0][0]
    cur_x1 = boxes[0][0] + boxes[0][2]
    for i in range(len(boxes) - 1):
        x, y, w, h = boxes[i]
        xn, yn, wn, hn = boxes[i + 1]
        gap = xn - (x + w)
        if gap <= thr:
            cur_x1 = max(cur_x1, xn + wn)
        else:
            tokens.append((cur_x0, cur_x1))
            cur_x0 = xn
            cur_x1 = xn + wn
    tokens.append((cur_x0, cur_x1))
    return tokens

# Remove a ghost first token based on geometry and brightness cues
def _drop_left_ghost_token(tokens: List[Tuple[int, int]], gray: np.ndarray) -> List[Tuple[int, int]]:
    if len(tokens) < 5:
        return tokens
    H, W = gray.shape[:2]
    widths = [x1 - x0 for (x0, x1) in tokens]
    medw = float(np.median(widths))
    gaps = [tokens[i + 1][0] - tokens[i][1] for i in range(len(tokens) - 1)]
    medg = float(np.median([g for g in gaps if g >= 0])) if gaps else 0.0

    x0, x1 = tokens[0]
    w0 = x1 - x0
    cx = 0.5 * (x0 + x1)
    roi = gray[:, max(0, x0):min(W, x1)]
    mean0 = float(np.mean(roi)) if roi.size else 255.0
    gap01 = gaps[0] if gaps else 0

    cond_far_left = (cx < 0.10 * W)
    cond_thin = (w0 < 0.70 * medw)
    cond_bright = (mean0 > 215.0)
    cond_isolated = (medg > 0 and gap01 > 1.7 * medg)

    if cond_far_left or cond_thin or cond_bright or cond_isolated:
        return tokens[1:]

    if len(tokens) >= 3:
        w1 = tokens[1][1] - tokens[1][0]
        w2 = tokens[2][1] - tokens[2][0]
        if (cx < 0.12 * W) and (w1 < 0.75 * medw) and (w2 >= 0.9 * medw):
            return tokens[1:]
    return tokens

# Divide a strip into n equal slots and return spans and rectangles
def equal_bins_slots(left2: np.ndarray, n: int = 6) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
    H, W = left2.shape[:2]
    gutter = max(4, int(0.01 * W))
    w_slot = W // n
    slots = []
    x = 0
    for i in range(n):
        x0 = max(0, x + gutter // 2)
        x1 = min(W, (x + w_slot) - gutter // 2)
        if i == 0:
            x0 = 0
        if i == n - 1:
            x1 = W
        slots.append((x0, x1))
        x += w_slot
    rects = [(x0, 0, x1 - x0, H) for (x0, x1) in slots]
    return slots, rects


# Locate the visual separator between the main numbers and the strong number
def locate_vertical_separator(row_gray: np.ndarray) -> int:
    h, w = row_gray.shape[:2]
    g = cv2.GaussianBlur(row_gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 10)
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 6)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kv, iterations=1)
    col_sum = np.sum(vert, axis=0)
    lx = int(0.60 * w)
    rx = int(0.90 * w)
    band = col_sum[lx:rx] if rx > lx else col_sum
    idx = np.argmax(band) if np.max(band) > 0 else np.argmax(col_sum)
    x_sep = int((lx + idx) if rx > lx else idx)
    return x_sep

# Detect a header band that shows fractions like 37/7
def is_fraction_header_band(band_bgr_or_gray: np.ndarray) -> bool:
    if band_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(band_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = band_bgr_or_gray
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    inv = 255 - thr
    cfg = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/|'
    txt = pytesseract.image_to_string(inv, config=cfg) or ""
    s = re.sub(r'\s+', '', txt)
    if '|' not in s or '/' not in s:
        return False
    m = re.search(r'(\d{1,2})/(\d{1,2})\|(\d{1,2})/(\d{1,2})', s)
    if not m:
        return False
    try:
        a, b, c, d = map(int, m.groups())
    except Exception:
        return False
    if b in (7, 37) and d in (7, 1):
        if 1 <= a <= 37 and 1 <= c <= 7:
            return True
    return False

# Detect top/bottom horizontal lines
def detect_horizontal_rules(gray: np.ndarray, min_length_ratio: float = 0.55, angle_tol_deg: float = 5.5) -> Optional[Tuple[int, int, float]]:
    h, w = gray.shape[:2]
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(g, 40, 120, apertureSize=3)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 7), 1))
    horiz = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)
    lines = cv2.HoughLinesP(horiz, 1, np.pi / 180, threshold=90,
                            minLineLength=int(min_length_ratio * w), maxLineGap=12)
    if lines is None:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=int(min_length_ratio * w), maxLineGap=10)
    if lines is None:
        return None
    ys = []
    for l in lines.reshape(-1, 4):
        x1, y1, x2, y2 = l
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        ang = math.degrees(math.atan2(dy, dx))
        if abs(ang) <= angle_tol_deg:
            ys.append((min(y1, y2), max(y1, y2), (y1 + y2) // 2, ang))
    if not ys:
        return None
    ys_sorted = sorted(ys, key=lambda t: t[2])
    top = ys_sorted[0]
    bot = ys_sorted[-1]
    return int(top[2]), int(bot[2]), float((top[3] + bot[3]) / 2.0)

# Segment row bands inside a ROI 
def segment_rows_in_roi(bin_img: np.ndarray, min_row_height: int = 12) -> List[Tuple[int, int]]:
    h, _ = bin_img.shape[:2]
    proj = np.sum(255 - bin_img, axis=1).astype(np.float32)
    proj = cv2.GaussianBlur(proj.reshape(-1, 1), (1, 9), 0).ravel()
    mx = float(np.max(proj)) if proj.size else 0.0
    thr = mx * 0.09 if mx > 0 else 0.0

    def scan_with(th):
        bands = []
        in_band = False
        start = 0
        for y in range(h):
            if proj[y] > th and not in_band:
                in_band, start = True, y
            elif proj[y] <= th and in_band:
                end = y
                if end - start >= min_row_height:
                    bands.append((start, end))
                in_band = False
        if in_band:
            end = h - 1
            if end - start >= min_row_height:
                bands.append((start, end))
        return bands

    bands = scan_with(thr) or scan_with(mx * 0.06 if mx > 0 else 0.0)
    if not bands:
        return []
    heights = [b[1] - b[0] for b in bands]
    med = int(np.median(heights)) if heights else 0
    out = []
    for (y1, y2) in bands:
        hgt = y2 - y1
        if med > 0 and hgt > 1.45 * med:
            local = bin_img[y1:y2]
            p2 = np.sum(255 - local, axis=1).astype(np.float32)
            p2 = cv2.GaussianBlur(p2.reshape(-1, 1), (1, 7), 0).ravel()
            m2 = float(np.max(p2)) if p2.size else 0.0
            th2 = m2 * 0.18 if m2 > 0 else 0.0
            st = None
            for k in range(local.shape[0]):
                if p2[k] > th2 and st is None:
                    st = k
                elif p2[k] <= th2 and st is not None:
                    en = k
                    if en - st >= min_row_height:
                        out.append((y1 + st, y1 + en))
                    st = None
            if st is not None:
                en = local.shape[0] - 1
                if en - st >= min_row_height:
                    out.append((y1 + st, y1 + en))
            continue
        out.append((y1, y2))
    return out

# Detect rows by tracking a vertical marker column on the left
def detect_rows_with_marker(gray: np.ndarray) -> List[Tuple[int,int]]:
    g = cv2.GaussianBlur(gray, (3,3), 0)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(g)
    bw = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    h, w = bw.shape
    xR = int(w*0.35)
    strip = bw[:, :xR]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(strip, connectivity=8)
    hs = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 10]
    if not hs:
        return []
    mh = float(np.median(hs))
    comps = []
    cx_vals = []
    for i in range(1, num_labels):
        y = stats[i, cv2.CC_STAT_TOP]
        hh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 10:
            continue
        if hh < 0.6*mh or hh > 1.6*mh:
            continue
        cx, cy = centroids[i]
        comps.append((int(y), int(y+hh), float(cx)))
        cx_vals.append(cx)
    if not comps:
        return []
    cx_med = float(np.median(cx_vals))
    rows = [(y0,y1) for (y0,y1,cx) in comps if abs(cx - cx_med) <= 6.0]
    rows.sort()
    merged = []
    for y0,y1 in rows:
        if not merged or y0 > merged[-1][1] + 3:
            merged.append([y0,y1])
        else:
            merged[-1][1] = max(merged[-1][1], y1)
    bands = [(int(a), int(b)) for a,b in merged]
    if len(bands) > 14:
        bands = sorted(bands, key=lambda t: t[1]-t[0], reverse=True)[:14]
        bands = sorted(bands)
    return bands

# Coarsely find candidate row bands from vertical projection over the page
def initial_bands(th: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    h, w = th.shape
    proj = th.sum(axis=1) // 255
    win = max(5, h // 200)
    if win > 1:
        kern = np.ones(win, dtype=np.float32) / win
        proj = np.convolve(proj, kern, mode="same")
    t = max(8, int(w * 0.06))
    mask = proj > t
    bands, i = [], 0
    while i < h:
        if mask[i]:
            s = i
            while i < h and mask[i]:
                i += 1
            e = i
            bands.append((s, e))
        i += 1
    merged = []
    for s, e in bands:
        if not merged:
            merged.append([s, e])
        else:
            ps, pe = merged[-1]
            if s - pe <= max(2, h // 200):
                merged[-1][1] = e
            else:
                merged.append([s, e])
    bands = [(s, e) for s, e in merged if (e - s) >= max(8, h // 50)]
    return bands, proj

# Refine and split oversized bands 
def refine_bands(th: np.ndarray, proj: np.ndarray, bands: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    heights = np.array([e - s for s, e in bands], dtype=int)
    if len(heights) == 0:
        return bands
    if len(heights) > 2:
        low, high = np.percentile(heights, [10, 90])
        core = heights[(heights >= low) & (heights <= high)]
        med = int(np.median(core)) if core.size else int(np.median(heights))
    else:
        med = int(np.median(heights))
    med = max(med, 20)
    refined = []
    for (s, e) in bands:
        if (e - s) <= int(1.6 * med):
            refined.append((s, e))
            continue
        sub = proj[s:e].astype(np.float32)
        if sub.size == 0:
            continue
        win = max(5, (e - s) // 40)
        if win > 1:
            kern = np.ones(win, dtype=np.float32) / win
            sub_s = np.convolve(sub, kern, mode="same")
        else:
            sub_s = sub
        mval = sub_s.max() if sub_s.size else 1.0
        thresh = mval * 0.45
        mins = np.r_[False, (sub_s[1:-1] < sub_s[:-2]) & (sub_s[1:-1] <= sub_s[2:]), False]
        cand = np.where(mins & (sub_s < thresh))[0]
        if cand.size == 0:
            parts = max(2, int(round((e - s) / med)))
            step = (e - s) // parts
            cuts = [s + step * i for i in range(1, parts)]
        else:
            min_gap = max(12, med // 2)
            cuts, prev = [], None
            for c in cand:
                y = s + int(c)
                if prev is None or (y - prev) >= min_gap:
                    cuts.append(y)
                    prev = y
        cuts = [y for y in cuts if (y - s) >= 8 and (e - y) >= 8]
        if not cuts:
            refined.append((s, e))
        else:
            ys = [s] + cuts + [e]
            for i in range(len(ys) - 1):
                a, b = ys[i], ys[i + 1]
                if (b - a) >= 10:
                    refined.append((a, b))
    return refined

# Filter bands by relative height and ink density 
def filter_bands(th: np.ndarray, bands: List[Tuple[int, int]], _img_h: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    if not bands:
        return bands, []
    heights = np.array([e - s for s, e in bands], dtype=int)
    med = int(np.median(heights))
    keep, rej = [], []
    h, w = th.shape
    x0, x1 = int(w * 0.12), int(w * 0.88)
    for idx, (s, e) in enumerate(bands):
        bh = e - s
        roi = th[s:e, x0:x1]
        density = float(roi.sum()) / (255.0 * max(1, roi.size))
        is_bottom = (idx == len(bands) - 1)
        low_h = int(0.60 * med) if not is_bottom else int(0.45 * med)
        low_d = 0.03 if not is_bottom else 0.02
        if idx == 0 and bh < int(0.90 * med):
            rej.append((s, e))
            continue
        if bh < low_h or bh > int(1.90 * med):
            rej.append((s, e))
            continue
        if density < low_d:
            rej.append((s, e))
            continue
        keep.append((s, e))
    return keep, rej

# Tighten the vertical bounds of a candidate row by trimming to scanlines where horizontal projection exceeds a central-width threshold
def tighten_band(th: np.ndarray, s: int, e: int) -> Tuple[int, int]:
    h, w = th.shape
    x0, x1 = int(w * 0.06), int(w * 0.94)
    roi = th[s:e, x0:x1]
    if roi.size == 0:
        return (s, e)
    proj = roi.sum(axis=1) // 255
    width = (x1 - x0)
    thr = max(2, int(width * (0.006 if width <= 800 else 0.012)))
    ys = np.where(proj > thr)[0]
    if ys.size >= 2:
        ns = s + int(ys[0])
        ne = s + int(ys[-1]) + 1
        return (ns, ne)
    return (s, e)

# Split overly tall bands into two; return new spans, median height, and split events.
def split_fat_spans_in_roi(gray: np.ndarray, spans: List[Tuple[int, int]], y_base: int, min_row_height: int = 12) -> Tuple[List[Tuple[int, int]], int, List[Tuple[int, int, int]]]:
    if not spans:
        return spans, 0, []
    heights = [e - s for (s, e) in spans]
    med_h = int(np.median(heights)) if heights else 0
    if med_h <= 0:
        return spans, 0, []
    fat_ratio = 1.5
    min_seg = max(min_row_height, int(0.60 * med_h))
    out: List[Tuple[int, int]] = []
    events: List[Tuple[int, int, int]] = []
    for (s, e) in spans:
        h = e - s
        if h >= int(fat_ratio * med_h):
            local = gray[y_base + s:y_base + e]
            th = binarize_for_lines(local)
            proj = th.sum(axis=1).astype(np.float32)
            if proj.size >= 5:
                k = max(5, int(h // 40) | 1)
                sm = cv2.GaussianBlur(proj.reshape(-1, 1), (1, k), 0).ravel()
                lo = int(0.33 * h)
                hi = int(0.67 * h)
                lo = max(1, min(lo, h - 2))
                hi = max(lo + 1, min(hi, h - 1))
                seg = sm[lo:hi]
                if seg.size > 0:
                    idx = int(np.argmin(seg)) + lo
                    a = s
                    b = s + idx
                    c = e
                    if (b - a) >= min_seg and (c - b) >= min_seg:
                        out.append((a, b))
                        out.append((b, c))
                        events.append((s, e, b))
                        continue
        out.append((s, e))
    return out, med_h, events