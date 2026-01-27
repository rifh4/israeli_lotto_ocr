import os
import re
import glob
import cv2
import numpy as np
import pytesseract
from typing import Dict, List, Optional, Tuple, Sequence

# Imports from our new modules
from config import (
    NUM_MIN, NUM_MAX,
    TEMPLATE_STRONG, TEMPLATE_OK,
    NUM_TEMPLATE_STRONG
)
from image_utils import _prep_for_match

# --- GLOBAL STATE FOR TEMPLATES ---
_TEMPLATES: Dict[int, List[np.ndarray]] = {}
_TEMPLATES_READY = False
_TEMPLATES_ROOT_PRINTED = False

_NUM_TEMPLATES: Dict[int, List[np.ndarray]] = {}
_NUM_TEMPLATES_READY = False

# Decide if a number template match score is strong enough
def _num_template_accepts(conf: float) -> bool:
    try:
        return float(conf) >= NUM_TEMPLATE_STRONG
    except Exception:
        return False

# Locate the folder containing digit templates (0–9)
def _digits_root() -> str:
    candidates = [
        os.getenv("DIGITS_DIR", "").strip(),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Digits"),
        os.path.join(os.getcwd(), "Digits"),
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return ""

# Locate the folder containing two-digit number templates (01–37)
def _numbers_root() -> str:
    c = os.getenv("LOTTO_NUMBERS_DIR", "").strip()
    if c and os.path.isdir(c):
        return c
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "lotto_numbers"),
        os.path.join(here, "Numbers"),
        os.path.join(here, "numbers"),
        "lotto_numbers",
        "Numbers",
        "numbers",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return here

# Yield (digit, path) pairs by scanning a directory tree for digit images
def _iter_digit_images(root: str):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(exts):
                continue
            d = None
            b = os.path.basename(fn)
            if b and b[0].isdigit():
                d = int(b[0])
            else:
                parent = os.path.basename(dirpath)
                if len(parent) == 1 and parent.isdigit():
                    d = int(parent)
            if d is None or not (0 <= d <= 9):
                continue
            yield d, os.path.join(dirpath, fn)

# Load digit (0–9) templates one time and report counts
def _load_digit_templates_once() -> None:
    global _TEMPLATES_READY, _TEMPLATES, _TEMPLATES_ROOT_PRINTED
    if _TEMPLATES_READY:
        return
    root = _digits_root()
    if not _TEMPLATES_ROOT_PRINTED:
        print(f"[digits] root: {root if root else '(not found)'}")
        _TEMPLATES_ROOT_PRINTED = True
    _TEMPLATES = {i: [] for i in range(10)}
    if root:
        examples: Dict[int, Optional[str]] = {i: None for i in range(10)}
        for d, p in _iter_digit_images(root):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Note: _prep_for_match is imported from image_utils
            _TEMPLATES[d].append(_prep_for_match(img))
            if examples[d] is None:
                examples[d] = p
        print("templates loaded (digit: count):")
        for d in range(10):
            ex = f"  e.g. ...\\{os.path.basename(examples[d])}" if examples[d] else ""
            print(f"  {d}: {len(_TEMPLATES[d])}{ex}")
    else:
        print("templates loaded (digit: count):")
        for d in range(10):
            print(f"  {d}: 0")
    print(f"TEMPLATE_STRONG={TEMPLATE_STRONG} TEMPLATE_OK={TEMPLATE_OK}")
    if sum(len(v) for v in _TEMPLATES.values()) == 0:
        print("[warn] no digit templates found")
    _TEMPLATES_READY = True

# Load 1–37 number templates one time
def _ensure_num_templates_loaded() -> None:
    global _NUM_TEMPLATES_READY, _NUM_TEMPLATES
    if _NUM_TEMPLATES_READY:
        return
    root = _numbers_root()
    d: Dict[int, List[np.ndarray]] = {i: [] for i in range(1, 38)}
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(root, e)))
    for i in range(1, 38):
        sub = os.path.join(root, f"{i:02d}")
        if os.path.isdir(sub):
            for e in exts:
                files.extend(glob.glob(os.path.join(sub, e)))
    for f in files:
        b = os.path.basename(f)
        m = re.match(r"^(\d{1,2})", b)
        if not m:
            continue
        n = int(m.group(1))
        if 1 <= n <= 37:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            d[n].append(th)
    _NUM_TEMPLATES = d
    _NUM_TEMPLATES_READY = True

# Match a probe image against all digit templates and return best digit and score
def _match_templates(img: np.ndarray) -> Tuple[Optional[int], float]:
    _load_digit_templates_once()
    probe = _prep_for_match(img)
    best_d, best_s = None, -1.0
    if not any(_TEMPLATES.values()):
        return None, -1.0
    for d, templs in _TEMPLATES.items():
        for t in templs:
            s = cv2.matchTemplate(probe, t, cv2.TM_CCOEFF_NORMED)
            v = float(s.max()) if s is not None else -1.0
            if v > best_s:
                best_s = v
                best_d = d
    return best_d, best_s

# Compute best template score for a specific digit
def _best_score_for_digit(img: np.ndarray, d: int) -> float:
    _load_digit_templates_once()
    templs = _TEMPLATES.get(d, [])
    if not templs:
        return -1.0
    probe = _prep_for_match(img)
    best = -1.0
    for t in templs:
        s = cv2.matchTemplate(probe, t, cv2.TM_CCOEFF_NORMED)
        v = float(s.max()) if s is not None else -1.0
        if v > best:
            best = v
    return best

# Match a cell against 1–37 templates within an allowed range
def _best_number_from_templates(cell_img: np.ndarray, allowed_min: int = 1, allowed_max: int = 37) -> Tuple[Optional[int], float]:
    _ensure_num_templates_loaded()
    if cell_img is None:
        return None, 0.0
    if len(cell_img.shape) == 3:
        g = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        g = cell_img
    _, binimg = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = binimg.shape[:2]
    best_n: Optional[int] = None
    best_s = -1.0
    for n, tmpls in _NUM_TEMPLATES.items():
        if n < allowed_min or n > allowed_max:
            continue
        for t in tmpls:
            th, tw = t.shape[:2]
            if th != h or tw != w:
                t2 = cv2.resize(t, (w, h), interpolation=cv2.INTER_AREA)
            else:
                t2 = t
            res = cv2.matchTemplate(binimg, t2, cv2.TM_CCOEFF_NORMED)
            s = float(res.max())
            if s > best_s:
                best_s = s
                best_n = n
    if best_s < 0:
        return None, 0.0
    return best_n, best_s

# Run Tesseract and return the longest numeric string
def ocr_text_multi(gray: np.ndarray, whitelist: str, psms: Sequence[int]) -> str:
    out = ""
    for p in psms:
        cfg = f"-c tessedit_char_whitelist={whitelist} --oem 3 --psm {p}"
        s = pytesseract.image_to_string(gray, config=cfg) or ""
        s = s.strip().replace("\n", " ").replace("\r", " ")
        if len(s) > len(out):
            out = s
    return out

# Normalize common OCR confusions to digits and strip non-digits
def normalize_digits(s: str) -> str:
    trans = str.maketrans({'O': '0', 'o': '0', 'I': '1', 'l': '1', '|': '1', 'S': '5', 's': '5', 'B': '8'})
    s = s.translate(trans)
    return "".join(ch for ch in s if ch.isdigit())

# Detect whether a binary blob contains an inner hole (for 0,6,8,9)
def _has_hole(bw: np.ndarray) -> bool:
    cnts, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return False
    for i, _ in enumerate(cnts):
        if hier[0][i][2] != -1:
            return True
    return False

# Split a token crop into two halves using connected components or column minima
def split_token_by_cc(crop_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    H, W = crop_gray.shape[:2]
    g = cv2.GaussianBlur(crop_gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 8)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h < int(0.45 * H) or h > int(0.98 * H):
            continue
        if w < int(0.02 * W):
            continue
        boxes.append((x, y, w, h))
    if boxes:
        boxes.sort(key=lambda r: r[0])
        pad = max(2, int(0.02 * W))
        if len(boxes) == 1:
            x, y, w, h = boxes[0]
            mid = x + (w // 2)
            lx0 = max(0, x - pad)
            lx1 = min(W, mid)
            rx0 = max(0, mid)
            rx1 = min(W, x + w + pad)
            left = crop_gray[:, lx0:lx1]
            right = crop_gray[:, rx0:rx1]
            return left, right, left.shape[1], right.shape[1]
        if len(boxes) == 2:
            x0, w0 = boxes[0][0], boxes[0][2]
            x1, w1 = boxes[1][0], boxes[1][2]
            lx0 = max(0, x0 - pad)
            lx1 = min(W, x0 + w0 + pad)
            rx0 = max(0, x1 - pad)
            rx1 = min(W, x1 + w1 + pad)
            return crop_gray[:, lx0:lx1], crop_gray[:, rx0:rx1], (lx1 - lx0), (rx1 - rx0)
        gaps: List[Tuple[int, int]] = []
        for i in range(1, len(boxes)):
            x_prev, y_prev, w_prev, h_prev = boxes[i - 1]
            x_cur, y_cur, w_cur, h_cur = boxes[i]
            gap = x_cur - (x_prev + w_prev)
            gaps.append((i, gap))
        best_i, best_score = 1, -1e9
        total_w = sum(w for (_, _, w, _) in boxes)
        for i, gap in gaps:
            left_group = boxes[:i]
            right_group = boxes[i:]
            lx0 = left_group[0][0]
            lx1 = left_group[-1][0] + left_group[-1][2]
            rx0 = right_group[0][0]
            rx1 = right_group[-1][0] + right_group[-1][2]
            lw = lx1 - lx0
            rw = rx1 - rx0
            bal = 1.0 - abs(lw - rw) / max(1.0, lw + rw)
            density = (lw + rw) / max(1.0, total_w)
            score = float(gap) * 1.0 + 40.0 * bal + 10.0 * density
            if score > best_score:
                best_score = score
                best_i = i
        left_group = boxes[:best_i]
        right_group = boxes[best_i:]
        lx0 = max(0, left_group[0][0] - pad)
        lx1 = min(W, left_group[-1][0] + left_group[-1][2] + pad)
        rx0 = max(0, right_group[0][0] - pad)
        rx1 = min(W, right_group[-1][0] + right_group[-1][2] + pad)
        if (lx1 - lx0) >= int(0.06 * W) and (rx1 - rx0) >= int(0.06 * W):
            return crop_gray[:, lx0:lx1], crop_gray[:, rx0:rx1], (lx1 - lx0), (rx1 - rx0)
    col = (bw > 0).sum(axis=0).astype(np.float32)
    if col.size >= 5:
        k = max(3, (W // 40) | 1)
        col = cv2.GaussianBlur(col.reshape(-1, 1), (k, 1), 0).ravel()
    lo, hi = int(0.12 * W), int(0.88 * W)
    split = int(np.argmin(col[lo:hi]) + lo) if hi > lo else W // 2
    split = max(int(0.08 * W), min(W - int(0.08 * W), split))
    left = crop_gray[:, :split]
    right = crop_gray[:, split:]
    return left, right, split, W - split

# Read a single digit from a half-token using contour crop + templates
def _digit_from_half(half_gray: np.ndarray) -> Tuple[Optional[int], Dict[str, float]]:
    g = cv2.GaussianBlur(half_gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        d, s = _match_templates(half_gray)
        if s < TEMPLATE_OK:
            d = None
        return d, {"source": "template", "confidence": 0.0, "tmpl_digit": d if d is not None else -1, "tmpl_score": s}
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    roi = half_gray[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2]
    d, s = _match_templates(roi)
    if s < TEMPLATE_OK:
        d = None
    return d, {"source": "template", "confidence": 0.0, "tmpl_digit": d if d is not None else -1, "tmpl_score": s}

# OCR a whole token as up to two digits and clamp to valid range
def ocr_token_whole(gray: np.ndarray) -> Optional[int]:
    s = normalize_digits(ocr_text_multi(gray, "0123456789", (7, 6, 10)))
    if not s:
        return None
    s = s[:2]
    v = int(s) if s.isdigit() else None
    if v is None:
        return None
    return v if NUM_MIN <= v <= NUM_MAX else None

# Convenience wrapper to read one digit from a small crop
def read_single_digit(gray: np.ndarray) -> Optional[int]:
    d, _ = _digit_from_half(gray)
    return d

# Reconcile whole/pair readings using 1–37 template matches
def resolve_with_number_templates(cell_img: np.ndarray,
                                  whole_num: Optional[int], whole_conf: float,
                                  pair_num: Optional[int], pair_conf: float,
                                  allowed_min: int = 1, allowed_max: int = 37) -> Tuple[Optional[int], str, Optional[int], float]:
    num_n, num_s = _best_number_from_templates(cell_img, allowed_min, allowed_max)
    if num_n is not None and _num_template_accepts(num_s):
        return num_n, "num_tmpl_strong", num_n, num_s
    if num_n is not None and pair_num is not None and num_n == pair_num:
        return pair_num, "pair_agrees_num", num_n, num_s
    if num_n is not None and whole_num is not None and num_n == whole_num:
        return whole_num, "whole_agrees_num", num_n, num_s
    if pair_num is not None and (pair_conf >= whole_conf):
        return pair_num, "pair_fallback", num_n, num_s
    return whole_num, "whole_fallback", num_n, num_s

# Read a two-digit slot using whole OCR, per-half templates, and heuristics
def read_slot_variants(crop_gray: np.ndarray) -> Tuple[Optional[int], Dict[str, str]]:
    dL_raw, dR_raw, wL, wR = split_token_by_cc(crop_gray)
    if dL_raw.size == 0 or dR_raw.size == 0:
        return None, {"whole": "", "a": "", "b": "", "pair": "", "decision": "empty_halves", "a_src": "", "a_conf": "0.00", "a_tmpl": "-1@-1.00", "b_src": "", "b_conf": "0.00", "b_tmpl": "-1@-1.00", "wL": str(wL), "wR": str(wR)}
    dL = cv2.resize(dL_raw, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    dR = cv2.resize(dR_raw, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    whole_raw = ocr_text_multi(crop_gray, "0123456789", (7, 6, 10, 8, 13))
    whole = normalize_digits(whole_raw)
    whole_val = int(whole[:2]) if len(whole) >= 2 and whole[:2].isdigit() else (int(whole) if whole.isdigit() else None)
    if whole_val is not None and not (NUM_MIN <= whole_val <= NUM_MAX):
        whole_val = None
    a, dbg_a = _digit_from_half(dL)
    b, dbg_b = _digit_from_half(dR)
    pair_val = a * 10 + b if (a is not None and b is not None) else None
    pair_val = pair_val if (pair_val is not None and NUM_MIN <= pair_val <= NUM_MAX) else None
    gR = cv2.GaussianBlur(dR, (3, 3), 0)
    bwR = cv2.adaptiveThreshold(gR, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 8)
    bwR = cv2.morphologyEx(bwR, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), 1)
    bwR = cv2.morphologyEx(bwR, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1)), 1)
    hole = _has_hole(bwR)
    s0 = _best_score_for_digit(dR, 0)
    s6 = _best_score_for_digit(dR, 6)
    s7 = _best_score_for_digit(dR, 7)
    s9 = _best_score_for_digit(dR, 9)
    s4 = _best_score_for_digit(dR, 4)
    if b == 7:
        cond_whole_6 = (whole_val is not None and a is not None and (whole_val // 10) == a and (whole_val % 10) == 6)
        if (hole and s6 >= s7 + 0.05) or cond_whole_6:
            if a is not None:
                b = 6
                pv = a * 10 + 6
                pair_val = pv if NUM_MIN <= pv <= NUM_MAX else pair_val
    if b == 6:
        cond_whole_7 = (whole_val is not None and a is not None and (whole_val // 10) == a and (whole_val % 10) == 7)
        if ((not hole) and s7 >= s6 + 0.05) or cond_whole_7:
            if a is not None:
                b = 7
                pv = a * 10 + 7
                pair_val = pv if NUM_MIN <= pv <= NUM_MAX else pair_val
    both_templates_strong = (dbg_a.get("tmpl_score", 0) >= max(TEMPLATE_STRONG, 0.86) and dbg_b.get("tmpl_score", 0) >= max(TEMPLATE_STRONG, 0.86))
    one_strong_one_ok = ((dbg_a.get("tmpl_score", 0) >= max(TEMPLATE_STRONG, 0.86) and dbg_b.get("tmpl_score", 0) >= TEMPLATE_OK) or (dbg_b.get("tmpl_score", 0) >= max(TEMPLATE_STRONG, 0.86) and dbg_a.get("tmpl_score", 0) >= TEMPLATE_OK))
    any_template_ok = (dbg_a.get("tmpl_score", 0) >= TEMPLATE_OK) or (dbg_b.get("tmpl_score", 0) >= TEMPLATE_OK)
    whole_is_single = (whole.isdigit() and len(whole) == 1)
    if pair_val is None and whole_is_single and whole != "1":
        best_d, best_s = None, -1.0
        for d in range(10):
            sc = _best_score_for_digit(dR, d)
            if sc > best_s:
                best_s = sc
                best_d = d
        if best_d is not None:
            pv = int(whole) * 10 + best_d
            if NUM_MIN <= pv <= NUM_MAX:
                b = best_d
                pair_val = pv
    chosen, decision = None, ""
    if a == 1:
        width_ratio = (wL / float(wR)) if (wL > 0 and wR > 0) else 1.0
        alt_best = max(s9, s7, s4, s6)
        allow_10 = (((whole_val == 10) and len(whole) >= 2) or (width_ratio < 0.72)) and (s0 >= alt_best + 0.12)
        if b in (0, 4, 6, 7, 9):
            if allow_10:
                chosen, decision = 10, "strict_10"
            elif b == 0:
                if alt_best > -1.0:
                    nb = 9 if s9 == alt_best else (7 if s7 == alt_best else (4 if s4 == alt_best else 6))
                    pv = 10 + nb
                    if NUM_MIN <= pv <= NUM_MAX:
                        b = nb
                        pair_val = pv
    if chosen is None and whole_val is not None and pair_val is not None and pair_val != whole_val:
        left_is_slim = (wL > 0 and wR > 0 and wL < 0.80 * wR)
        if (whole_val // 10) == 1 and left_is_slim:
            chosen, decision = whole_val, "whole_pref_slim_left"
    if chosen is None:
        if pair_val is not None and (both_templates_strong or one_strong_one_ok) and (whole_val is None or pair_val == whole_val):
            chosen, decision = pair_val, "pair"
        elif whole_is_single and whole != "1" and pair_val is not None and any_template_ok:
            chosen, decision = pair_val, "pair_over_single_whole"
        elif whole_val is not None and pair_val is None:
            chosen, decision = whole_val, "whole"
        elif whole_val is not None and pair_val is not None:
            tens_equal = (whole_val // 10) == (pair_val // 10)
            ones_set = {whole_val % 10, pair_val % 10}
            if tens_equal and ones_set == {6, 7}:
                s6b = _best_score_for_digit(dR, 6)
                s7b = _best_score_for_digit(dR, 7)
                ones = 7 if s7b >= s6b - 0.02 else 6
                chosen, decision = (10 * (whole_val // 10) + ones), "ones_6_7_by_template"
            else:
                pair_conf = min(dbg_a.get("tmpl_score", 0.0), dbg_b.get("tmpl_score", 0.0))
                if pair_conf >= TEMPLATE_OK:
                    chosen, decision = pair_val, "pair_conf_ok"
                else:
                    chosen, decision = pair_val, "pair_conf_low"
        else:
            chosen, decision = pair_val if pair_val is not None else whole_val, "fallback"
    pair_conf_final = min(dbg_a.get("tmpl_score", 0.0), dbg_b.get("tmpl_score", 0.0)) if (a is not None and b is not None) else 0.0
    chosen2, decision2, _, _ = resolve_with_number_templates(crop_gray, whole_val, 0.0, pair_val, pair_conf_final, 1, 37)
    if chosen2 is not None:
        chosen, decision = chosen2, decision2
    dbg = {"whole": ("" if whole_val is None else str(whole_val)), "a": ("" if a is None else str(a)), "b": ("" if b is None else str(b)), "pair": ("" if pair_val is None else str(pair_val)), "decision": decision, "a_src": dbg_a.get("source", ""), "a_conf": f"{dbg_a.get('confidence', 0.0):.2f}", "a_tmpl": f"{dbg_a.get('tmpl_digit', -1)}@{dbg_a.get('tmpl_score', -1.0):.2f}", "b_src": dbg_b.get("source", ""), "b_conf": f"{dbg_b.get('confidence', 0.0):.2f}", "b_tmpl": f"{dbg_b.get('tmpl_digit', -1)}@{dbg_b.get('tmpl_score', -1.0):.2f}", "wL": str(wL), "wR": str(wR)}
    return chosen, dbg