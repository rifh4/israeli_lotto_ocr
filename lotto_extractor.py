import re
import cv2
import numpy as np
import pytesseract
import concurrent.futures
from typing import List, Dict, Optional, Tuple

# Imports from our new modules
from config import (
    NUM_MIN, NUM_MAX, STR_MIN, STR_MAX,
    TEMPLATE_OK, ENABLE_AI,
    AI_MODEL_READ, AI_TIMEOUT_MS
)
from image_utils import (
    normalize_row_height,
    clip_with_min_pad,
    binarize_for_lines
)
from ocr_engine import (
    _match_templates,
    _best_score_for_digit,
    _has_hole,
    read_slot_variants,
    read_single_digit,
    ocr_token_whole
)
from ai_client import (
    ai_read_numbers_from_left,
    ai_confirm_and_merge_with_read,
    ai_verify_numbers_in_left,
    _safe_invariants_ok
)
from layout_analysis import (
    cc_digit_boxes,
    merge_boxes_to_tokens,
    _drop_left_ghost_token,
    equal_bins_slots,
    locate_vertical_separator,
    is_fraction_header_band,
    detect_horizontal_rules,
    detect_rows_with_marker,
    segment_rows_in_roi,
    initial_bands,
    refine_bands,
    filter_bands,
    tighten_band,
    split_fat_spans_in_roi
)

# --- GLOBAL CONSTANTS ---
# Loaded once to avoid repeated OS calls in loops
AI_MODEL = AI_MODEL_READ
AI_TIMEOUT = AI_TIMEOUT_MS


# Read the strong number on the right using ROI heuristics and templates
def read_strong(right_img: np.ndarray) -> Optional[int]:
    H, W = right_img.shape[:2]
    
    def _parse_leading_outside_parens(s: str) -> Optional[int]:
        s0 = s.strip()
        m = re.search(r'^\s*(\d{1,2})\s*(?=[\(\[]|$)', s0)
        if m:
            v = int(m.group(1))
            return v if STR_MIN <= v <= STR_MAX else None
        s_no_par = re.sub(r'[\(\[][^\)\]]*[\)\]]', ' ', s0)
        m2 = re.search(r'\b(\d{1,2})\b', s_no_par)
        if m2:
            v = int(m2.group(1))
            return v if STR_MIN <= v <= STR_MAX else None
        return None
    
    def _ocr_strings(img: np.ndarray) -> Tuple[List[str], bool]:
        outs: List[str] = []
        has_par = False
        variants = [img, cv2.equalizeHist(img), 255 - cv2.equalizeHist(img)]
        for v in variants:
            for psm in (7, 6, 8, 11, 13):
                s = pytesseract.image_to_string(
                    v,
                    config=f"-c tessedit_char_whitelist=()0123456789 --oem 3 --psm {psm}"
                ) or ""
                s = s.strip()
                if s:
                    outs.append(s)
                    if "(" in s:
                        has_par = True
        seen = set()
        uniq = []
        for x in outs:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        return uniq, has_par
    
    def _left_bias_rois(img: np.ndarray):
        g = cv2.GaussianBlur(img, (3, 3), 0)
        bw0 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
        bw = cv2.morphologyEx(bw0, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = img.shape
        cand = []
        for c in contours:
            x, y, ww, hh = cv2.boundingRect(c)
            if hh < int(0.35 * h) or hh > int(0.95 * h):
                continue
            if ww < int(0.05 * w) or ww > int(0.65 * w):
                continue
            cand.append((x, y, ww, hh))
        rois: List[Tuple[str, np.ndarray, Tuple[int, int, int, int], Tuple[int, int, int]]] = []
        band_w = max(20, int(0.45 * w))
        rois.append(("leftBand45%", img[:, :band_w], (0, 0, band_w, h), (255, 255, 0)))
        if cand:
            cand.sort(key=lambda r: r[0])
            x, y, ww, hh = cand[0]
            pad = max(4, int(0.04 * h))
            lx0 = max(0, x - pad)
            lx1 = min(w, x + ww + pad)
            max_w = max(int(0.35 * w), 30)
            lx1 = min(lx1, lx0 + max_w)
            rois.append(("tightBlobLeft", img[:, lx0:lx1], (lx0, 0, lx1 - lx0, h), (0, 200, 0)))
            exp = min(w, int(lx1 + 0.10 * w))
            rois.append(("leftExpand", img[:, lx0:exp], (lx0, 0, exp - lx0, h), (255, 0, 255)))
        return rois

    rois = _left_bias_rois(right_img)

    chosen = None
    chosen_roi = None
    chosen_img = None
    max_s6_roi = -1.0

    for label, roi_img, _, _ in rois:
        td, _ = _match_templates(roi_img)
        s6_here = _best_score_for_digit(roi_img, 6)
        if s6_here > max_s6_roi:
            max_s6_roi = s6_here
        outs, has_par = _ocr_strings(roi_img)
        if td is not None and STR_MIN <= td <= STR_MAX:
            outs = [str(td)] + outs
        if label != "leftBand45%" and has_par:
            continue
        for s in outs:
            v = _parse_leading_outside_parens(s)
            if v is not None:
                chosen = v
                chosen_roi = label
                chosen_img = roi_img
                break
        if chosen is not None:
            break

    if chosen is None:
        left_half = right_img[:, :max(int(0.5 * W), 24)]
        best_d, best_s = None, -1.0
        for d in range(STR_MIN, STR_MAX + 1):
            sc = _best_score_for_digit(left_half, d)
            if sc > best_s:
                best_s = sc
                best_d = d
        if best_d is not None and best_s >= TEMPLATE_OK:
            chosen = best_d
            chosen_roi = "left_half_fallback"
            chosen_img = left_half

    if chosen in (3, 6) and chosen_img is not None:
        mid = chosen_img.shape[1] // 2
        dR = chosen_img[:, mid:]
        g = cv2.GaussianBlur(dR, (3, 3), 0)
        bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 8)
        hole = _has_hole(bw)
        s3 = _best_score_for_digit(dR, 3)
        s6 = _best_score_for_digit(dR, 6)
        if chosen == 6:
            if max_s6_roi < 0.86 and (not hole) and (s3 >= s6 + 0.07) and (s6 < 0.82):
                chosen = 3
                chosen_roi = chosen_roi
        else:
            if hole and (s6 >= s3 + 0.02):
                chosen = 6
                chosen_roi = chosen_roi

    return chosen

# Extract up to six main numbers from the left region using CC + heuristics
def numbers_from_cc(left2: np.ndarray) -> List[int]:
    boxes = cc_digit_boxes(left2)
    tokens = merge_boxes_to_tokens(boxes)
    tokens = _drop_left_ghost_token(tokens, left2)

    vals: List[int] = []
    val_to_token: List[int] = []
    H, W = left2.shape[:2]

    for t_idx, (x0, x1) in enumerate(tokens, start=1):
        x0p, x1p = max(0, x0 - 2), min(W, x1 + 12)
        crop = left2[:, x0p:x1p]
        crop = cv2.resize(crop, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
        v, _ = read_slot_variants(crop)
        if v is not None:
            vals.append(v)
            val_to_token.append(t_idx - 1)
        if len(vals) >= 6:
            break

    if len(vals) >= 4:
        best_s, best_e, cur_s = 0, 0, 0
        for i in range(len(vals) - 1):
            if not (vals[i] > vals[i + 1]):
                if i - cur_s + 1 > best_e - best_s + 1:
                    best_s, best_e = cur_s, i
                cur_s = i + 1
        if len(vals) - 1 - cur_s + 1 > best_e - best_s + 1:
            best_s, best_e = cur_s, len(vals) - 1
        if best_e - best_s + 1 >= 4 and (best_s != 0 or best_e != len(vals) - 1):
            vals = vals[best_s:best_e + 1]
            val_to_token = val_to_token[best_s:best_e + 1]

    if len(vals) < 6 and len(tokens) > 6:
        spans, _ = equal_bins_slots(left2, 6)
        for i, (x0, x1) in enumerate(spans):
            if i == 0:
                continue
            crop = left2[:, x0:x1]
            crop = cv2.resize(crop, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
            v, _ = read_slot_variants(crop)
            if v is not None:
                vals.append(v)
                val_to_token.append(min(i, len(tokens) - 1))
            if len(vals) >= 6:
                break

    if len(vals) < 6:
        rx0 = int(W * 0.60)
        crop = left2[:, rx0:W]
        boxes2 = cc_digit_boxes(crop)
        tokens2 = merge_boxes_to_tokens(boxes2)
        if len(tokens2) >= 1:
            a0, a1 = tokens2[-1]
            ca, cb = max(0, a0 - 2), min(crop.shape[1], a1 + 2)
            sl = crop[:, ca:cb]
            v, _ = read_slot_variants(cv2.resize(sl, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC))
            if v is not None and NUM_MIN <= v <= NUM_MAX and v not in vals:
                vals.append(v)

    for i in range(len(vals)):
        if i >= len(val_to_token):
            break
        if vals[i] < 10:
            t = val_to_token[i]
            x0, x1 = tokens[t]
            midL = (tokens[t - 1][0] + tokens[t - 1][1]) // 2 if t - 1 >= 0 else 0
            midR = (tokens[t + 1][0] + tokens[t + 1][1]) // 2 if t + 1 < len(tokens) else W
            pad = 6
            x0p = max(midL, x0 - pad)
            x1p = min(midR, x1 + pad)
            crop = left2[:, x0p:x1p]
            boxes2 = cc_digit_boxes(crop)
            tokens2 = merge_boxes_to_tokens(boxes2)
            fixed = None
            if len(tokens2) == 2:
                tmp: List[Optional[int]] = []
                for (b0, b1) in tokens2:
                    ca, cb = max(0, b0 - 2), min(crop.shape[1], b1 + 2)
                    sl = crop[:, ca:cb]
                    d = read_single_digit(sl)
                    tmp.append(d)
                if tmp[0] is not None and tmp[1] is not None:
                    fixed = tmp[0] * 10 + tmp[1]
            if fixed is None:
                spans2, _ = equal_bins_slots(crop, 2)
                tmp: List[Optional[int]] = []
                for (b0, b1) in spans2:
                    sl = crop[:, b0:b1]
                    d = read_single_digit(sl)
                    tmp.append(d)
                if tmp[0] is not None and tmp[1] is not None:
                    fixed = tmp[0] * 10 + tmp[1]
            if fixed is not None and NUM_MIN <= fixed <= NUM_MAX:
                vals[i] = fixed

    vals = vals[:6]

    if len(vals) and len(val_to_token):
        for i in range(min(len(vals), len(val_to_token))):
            v = vals[i]
            if v < 10:
                continue
            t = val_to_token[i]
            if not (0 <= t < len(tokens)):
                continue
            x0, x1 = tokens[t]
            pad = 4
            x0p = max(0, x0 - pad)
            x1p = min(W, x1 + pad)
            token_crop = left2[:, x0p:x1p]
            token_crop = cv2.resize(token_crop, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
            mid = token_crop.shape[1] // 2
            dR = token_crop[:, mid:]
            gR = cv2.GaussianBlur(dR, (3, 3), 0)
            bwR = cv2.adaptiveThreshold(gR, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 8)
            hole = _has_hole(bwR)
            s6 = _best_score_for_digit(dR, 6)
            s7 = _best_score_for_digit(dR, 7)
            whole_hint = ocr_token_whole(token_crop)
            tens = v // 10
            ones = v % 10
            new_v = v
            if ones == 6:
                cond_hint7 = (whole_hint is not None and (whole_hint // 10) == tens and (whole_hint % 10) == 7)
                if (not hole and s7 >= s6 - 0.02) or cond_hint7:
                    cand = tens * 10 + 7
                    if NUM_MIN <= cand <= NUM_MAX:
                        if cand not in vals or vals.count(v) > 1:
                            new_v = cand
            elif ones == 7:
                cond_hint6 = (whole_hint is not None and (whole_hint // 10) == tens and (whole_hint % 10) == 6)
                if hole or cond_hint6 or (s6 >= s7 + 0.03):
                    cand = tens * 10 + 6
                    if NUM_MIN <= cand <= NUM_MAX:
                        if cand not in vals or vals.count(v) > 1:
                            new_v = cand
            vals[i] = new_v

    if len(vals) and len(val_to_token):
        for i in range(max(0, len(vals) - 2), len(vals)):
            if i >= len(val_to_token):
                continue
            t = val_to_token[i]
            if not (0 <= t < len(tokens)):
                continue
            x0, x1 = tokens[t]
            pad = 4
            x0p = max(0, x0 - pad)
            x1p = min(W, x1 + pad)
            token_crop = left2[:, x0p:x1p]
            token_crop = cv2.resize(token_crop, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
            mid = token_crop.shape[1] // 2
            dL = token_crop[:, :mid]
            s0 = _best_score_for_digit(dL, 0)
            if s0 >= TEMPLATE_OK:
                best_d, best_s = None, -1.0
                for d in range(1, 10):
                    sc = _best_score_for_digit(token_crop[:, mid:], d)
                    if sc > best_s:
                        best_s = sc
                        best_d = d
                if best_d is not None and NUM_MIN <= best_d <= NUM_MAX:
                    vals[i] = best_d

    return vals

# Sanitize a line: enforce ranges, de-dup, and collect warnings
def validate_line(numbers: List[int], strong: Optional[int]) -> Tuple[List[int], Optional[int], List[str]]:
    warnings: List[str] = []
    nums = [n for n in numbers if isinstance(n, int)]
    nums = [n for n in nums if NUM_MIN <= n <= NUM_MAX]
    if len(nums) != 6:
        warnings.append("too_few_in_range")
    if len(set(nums)) != len(nums):
        warnings.append("duplicates_in_main")
    s = strong if isinstance(strong, int) else None
    if s is None or not (STR_MIN <= s <= STR_MAX):
        warnings.append("strong_missing_or_out_of_range")
    return nums[:6], s, warnings

def apply_ai_slot_verification(left2: np.ndarray, numbers: List[int], verify_all_two_digit: bool = True) -> Tuple[List[int], Dict]:
    # Use global constants
    return ai_verify_numbers_in_left(left2, numbers, ambiguous_only=not verify_all_two_digit, model=AI_MODEL, timeout_ms=AI_TIMEOUT)

# Hybrid pipeline: classic read + API verify/read + final validation
def perfected_ocr(left2: np.ndarray, right2: np.ndarray) -> Tuple[List[int], Optional[int], List[str]]:
    nums_classic = numbers_from_cc(left2)
    strong = read_strong(right2)
    try:
        _, verify = apply_ai_slot_verification(left2, nums_classic, verify_all_two_digit=True)
    except Exception:
        verify = {"slots": []}
    
    # Simplified verify logic
    slots = verify.get("slots", [])
    if isinstance(slots, list):
        ai_verify_used = any(s.get("used") for s in slots if isinstance(s, dict))
    else:
        ai_verify_used = False

    # Use global constants
    ai_nums, meta = ai_read_numbers_from_left(left2, model=AI_MODEL, timeout_ms=AI_TIMEOUT)
    nums_merged, cinfo = ai_confirm_and_merge_with_read(nums_classic, verify, ai_nums, strong)
    
    ok_merged = _safe_invariants_ok(nums_merged)
    if ok_merged and nums_merged != nums_classic:
        nums3 = nums_merged
    elif not ai_verify_used and not _safe_invariants_ok(nums_classic) and isinstance(ai_nums, list) and len(ai_nums) == 6 and _safe_invariants_ok(ai_nums):
        nums3 = ai_nums
    else:
        nums3 = nums_classic

    final_numbers, final_strong, val_warnings = validate_line(nums3, strong)
    return final_numbers, final_strong, val_warnings

# Split a row into left (six numbers) and right (strong) regions, run classic+AI OCR, validate, and return a single line record.
def split_row_and_ocr(row_img: np.ndarray, idx: int) -> Dict:
    row_norm = normalize_row_height(row_img, 32)

    def _choose_cut(img: np.ndarray) -> Tuple[int, int]:
        H, W = img.shape[:2]
        guess = locate_vertical_separator(img)
        cands = [guess, int(0.74 * W), int(0.70 * W), int(0.66 * W), int(0.62 * W)]
        lo = int(0.58 * W)
        hi = int(0.85 * W)
        best = None
        for c in cands:
            if c is None:
                continue
            c = int(max(lo, min(hi, c)))
            lpad = max(2, int(0.020 * W))
            left = img[:, :max(0, c - lpad)]
            n = len(merge_boxes_to_tokens(cc_digit_boxes(left)))
            score = (-1.8 * abs(n - 6)) - (1.2 if n > 6 else 0.0) + (0.30 if 5 <= n <= 6 else -0.6)
            if (best is None) or (score > best[0]):
                best = (score, c, n)
        if best is None:
            return int(0.70 * W), 0
        return best[1], best[2]

    H, W = row_norm.shape
    cut, n_tokens = _choose_cut(row_norm)
    lpad = max(2, int(0.020 * W))
    left = row_norm[:, :max(0, cut - lpad)]
    right = row_norm[:, min(W, cut + lpad):]
    left2 = cv2.resize(left, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
    right2 = cv2.resize(right, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)

    nums_classic = numbers_from_cc(left2)
    strong = read_strong(right2)

    if (len(nums_classic) == 6 and len(set(nums_classic)) < 6):
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        left2_alt = cv2.morphologyEx(left2, cv2.MORPH_ERODE, k, iterations=1)
        nums2 = numbers_from_cc(left2_alt)
        if len(nums2) == 6 and len(set(nums2)) == 6:
            nums_classic = nums2

    nums3 = nums_classic
    if ENABLE_AI:
        try:
            _, verify = apply_ai_slot_verification(left2, nums_classic, verify_all_two_digit=True)
        except Exception:
            verify = {"slots": []}
        
        # Use global constants
        ai_nums, _ = ai_read_numbers_from_left(left2, model=AI_MODEL, timeout_ms=AI_TIMEOUT)
        nums_merged, _ = ai_confirm_and_merge_with_read(nums_classic, verify, ai_nums, strong)
        ok_merged = _safe_invariants_ok(nums_merged)
        if ok_merged and nums_merged != nums_classic:
            nums3 = nums_merged
        elif not _safe_invariants_ok(nums_classic) and isinstance(ai_nums, list) and len(ai_nums) == 6 and _safe_invariants_ok(ai_nums):
            nums3 = ai_nums

    final_numbers, final_strong, _ = validate_line(nums3, strong)
    rec = {
        "line_index": idx,
        "numbers": final_numbers if final_numbers else [],
        "strong": final_strong
    }
    return rec

# Read a single line ticket, OCR one row and return [record] only if it has 6 numbers and a valid strong
def extract_single_line(gray: np.ndarray, start_index: int = 1) -> List[Dict]:
    idx = int(start_index)
    try:
        rec = split_row_and_ocr(gray, idx)
        ok = len(rec.get("numbers") or []) >= 6 and rec.get("strong") is not None
        return [rec] if ok else []
    except Exception:
        return []

# Full multi-line extraction, detect row bands, guard headers - PARALLEL VERSION
def extract_all_lines(gray: np.ndarray, start_index: int = 1) -> List[Dict]:
    # Compute row-band candidates within a vertical slice , with a fallback pass on the raw gray ROI
    def _run_roi(a: int, b: int):
        roi_gray = gray[a:b]
        y_base = a
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        roi_eq = clahe.apply(roi_gray)
        th_inv = binarize_for_lines(roi_eq)
        roi_bin = 255 - th_inv
        spans = segment_rows_in_roi(roi_bin, min_row_height=12)
        if len(spans) < 10:
            th_inv2 = binarize_for_lines(roi_gray)
            spans2 = segment_rows_in_roi(255 - th_inv2, min_row_height=12)
            if len(spans2) > len(spans):
                spans = spans2
        return spans, y_base
    
    def _right_has_two_fractions(img: np.ndarray) -> bool:
        if img.size == 0:
            return False
        h, w = img.shape[:2]
        rb = img[:, max(0, int(0.58 * w)):]
        s = pytesseract.image_to_string(rb, config="--oem 3 --psm 7 -c tessedit_char_whitelist=()|/0123456789") or ""
        s = re.sub(r"\s+", "", s)
        return bool(re.search(r"^\D*?(\d{1,2})/(\d{1,2})\|(\d{1,2})/(\d{1,2})", s))
    
    def _left_token_count(img: np.ndarray) -> int:
        h, w = img.shape[:2]
        xsep = locate_vertical_separator(img)
        lpad = max(2, int(0.015 * w))
        left = img[:, :max(0, xsep - lpad)]
        return len(merge_boxes_to_tokens(cc_digit_boxes(left)))

    def _left_digit_clusters(img: np.ndarray) -> int:
        h, w = img.shape[:2]
        xsep = locate_vertical_separator(img)
        lpad = max(2, int(0.015 * w))
        left = img[:, :max(0, xsep - lpad)]
        s = pytesseract.image_to_string(left, config="--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789") or ""
        g = re.findall(r"\d+", s)
        return len([x for x in g if len(x) >= 1])

    def _should_drop_header(img: np.ndarray) -> bool:
        return (_left_token_count(img) <= 1) and _right_has_two_fractions(img) and (_left_digit_clusters(img) <= 2)

    def _interval_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        s1, e1 = a
        s2, e2 = b
        inter = max(0, min(e1, e2) - max(s1, s2))
        uni = max(e1, e2) - min(s1, s2)
        return (inter / float(uni)) if uni > 0 else 0.0

    def _dedup_merge(spans: List[Tuple[int, int]], iou_th: float = 0.30) -> List[Tuple[int, int]]:
        spans = [(int(s), int(e)) for (s, e) in spans if e > s]
        spans.sort(key=lambda t: t[0])
        out: List[Tuple[int, int]] = []
        for s, e in spans:
            merged = False
            for i, (ps, pe) in enumerate(out):
                if _interval_iou((s, e), (ps, pe)) >= iou_th:
                    ns, ne = min(ps, s), max(pe, e)
                    out[i] = (ns, ne)
                    merged = True
                    break
            if not merged:
                out.append((s, e))
        out.sort(key=lambda t: t[0])
        return out

    def _anchor_and_grow(spans: List[Tuple[int, int]], y_lo: int, y_hi: int) -> List[Tuple[int, int]]:
        if not spans:
            return spans
        heights = [e - s for (s, e) in spans]
        gaps = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]
        med_h = int(np.median(heights)) if heights else 0
        med_g = int(np.median([g for g in gaps if g > 0])) if gaps else 0
        if med_h <= 0:
            med_h = max(18, (y_hi - y_lo) // 20)
        if med_g <= 0:
            med_g = max(6, med_h // 2)
        step = med_h + med_g
        half = max(4, med_h // 6)

        def _hunt(center_y: int) -> Optional[Tuple[int, int]]:
            a = max(y_lo, center_y - half)
            b = min(y_hi, center_y + half)
            if b - a < 8:
                return None
            roi_gray = gray[a:b]
            if roi_gray.size == 0:
                return None
            th_inv = binarize_for_lines(roi_gray)
            roi_bin = 255 - th_inv
            proj = np.sum(255 - roi_bin, axis=1).astype(np.float32)
            if proj.size == 0:
                return None
            if proj.size >= 5:
                k = max(5, int(proj.size // 20) | 1)
                proj = cv2.GaussianBlur(proj.reshape(-1, 1), (1, k), 0).ravel()
            mx = float(np.max(proj)) if proj.size else 0.0
            thr = mx * 0.06 if mx > 0 else 0.0
            in_band, s0 = False, 0
            best = None
            for i in range(proj.size):
                if proj[i] > thr and not in_band:
                    in_band, s0 = True, i
                elif proj[i] <= thr and in_band:
                    e0 = i
                    if (e0 - s0) >= max(10, med_h // 2):
                        best = (s0, e0)
                        break
                    in_band = False
            if best is None and in_band:
                e0 = proj.size - 1
                if (e0 - s0) >= max(10, med_h // 2):
                    best = (s0, e0)
            if best is None:
                return None
            return (a + best[0], a + best[1])

        out = spans[:]
        top_s, _ = out[0]
        _, bot_e = out[-1]
        fail_up = 0
        fail_dn = 0
        for _ in range(6):
            cy = top_s - step
            if cy <= y_lo:
                break
            cand = _hunt(cy)
            if cand and max((_interval_iou(cand, t) for t in out), default=0.0) < 0.25:
                out.insert(0, cand)
                top_s = cand[0]
                fail_up = 0
            else:
                fail_up += 1
            if fail_up >= 2:
                break
        for _ in range(6):
            cy = bot_e + step
            if cy >= y_hi:
                break
            cand = _hunt(cy)
            if cand and max((_interval_iou(cand, t) for t in out), default=0.0) < 0.25:
                out.append(cand)
                bot_e = cand[1]
                fail_dn = 0
            else:
                fail_dn += 1
            if fail_dn >= 2:
                break
        return _dedup_merge(out, 0.35)

    results: List[Dict] = []
    header_skipped: List[Tuple[int, int]] = []
    bands_rej: List[Tuple[int, int]] = []
    spansA_abs: List[Tuple[int, int]] = []
    rules = detect_horizontal_rules(gray)
    if rules is not None:
        y_top, y_bot, _ = rules
        h_img = gray.shape[0]
        roi_y1 = max(0, y_top + 2)
        roi_y2 = min(h_img, y_bot - 2)
        spansA, y_baseA = _run_roi(roi_y1, roi_y2)
        sm = detect_rows_with_marker(gray)
        if sm:
            sm_roi = [(s - roi_y1, e - roi_y1) for (s, e) in sm if s >= roi_y1 and e <= roi_y2]
            if len(sm_roi) >= max(len(spansA), 10):
                spansA = sm_roi
        if spansA:
            heightsA = [e - s for (s, e) in spansA]
            med_row_hA = int(np.median(heightsA)) if heightsA else max(12, (roi_y2 - roi_y1) // 14)
            hb_y0 = roi_y1
            hb_y1 = min(roi_y2, roi_y1 + int(1.15 * med_row_hA))
            header_found = False
            if hb_y1 > hb_y0:
                band = gray[hb_y0:hb_y1]
                if band.size > 0 and is_fraction_header_band(band):
                    header_found = True
            spansA_abs = [((s + y_baseA), (e + y_baseA)) for (s, e) in spansA]
            if header_found:
                header_end = hb_y1 + 4
                header_skipped.append((hb_y0, min(h_img, header_end)))
                spansA_abs = [(s, e) for (s, e) in spansA_abs if ((s + e) // 2) > header_end]

    th = binarize_for_lines(gray)
    h, w = th.shape
    bands0, proj = initial_bands(th)
    bands1 = refine_bands(th, proj, bands0)
    bands1 = sorted(bands1, key=lambda x: x[0])

    small_img = (w <= 900) or (h <= 600)

    bands_kept, bands_rej = filter_bands(th, bands1, h)
    tight = [tighten_band(th, s, e) for (s, e) in bands_kept]
    tight = [(s, e) for (s, e) in tight if e > s]

    if (not tight) or (small_img and len(tight) < 2):
        x0_c, x1_c = int(w * 0.12), int(w * 0.88)
        dens = []
        for (s, e) in bands1:
            roi = th[s:e, x0_c:x1_c]
            d = float(roi.sum()) / (255.0 * max(1, roi.size))
            dens.append(((s, e), d))
        dens.sort(key=lambda t: t[1], reverse=True)
        fallback = [se for (se, _) in dens[:max(2, len(tight))] if se not in header_skipped]
        if fallback:
            tight = [tighten_band(th, s, e) for (s, e) in fallback]
            bands_rej = [b for b in bands_rej if b not in fallback]

    spans_split, med_row_hB, events = split_fat_spans_in_roi(gray, tight, 0, min_row_height=12)
    spansB_abs = spans_split[:]
    sm2 = detect_rows_with_marker(gray)
    if sm2 and len(sm2) >= max(len(spansB_abs), 10):
        spansB_abs = sm2

    cand_spans = _dedup_merge(spansA_abs + spansB_abs, 0.35)
    if cand_spans:
        y_lo, y_hi = 0, gray.shape[0]
        cand_spans = _anchor_and_grow(cand_spans, y_lo, y_hi)

    med_row_height_all = int(np.median([e - s for (s, e) in cand_spans])) if cand_spans else 0
    
    # --- PARALLEL WORKER FUNCTION ---
    def _process_single_span(args):
        # Unpack arguments
        j, y0, y1 = args
        
        # Crop the row image
        crop_gray, _ = clip_with_min_pad(gray, y0, y1, pad=2)
        if crop_gray.size == 0:
            return None

        # Logic for header dropping (top_header_flag)
        top_header_flag = (j == 0 and _should_drop_header(crop_gray))
        
        # Basic height check
        band_h = int(y1 - y0)
        min_h = max(20 if small_img else 24,
                    int((0.50 if small_img else 0.65) * med_row_height_all)) if med_row_height_all > 0 else (20 if small_img else 24)
        if band_h < min_h:
            return None

        # Perform OCR (this includes network calls if AI is on)
        rec = split_row_and_ocr(crop_gray, 0) # idx is 0 for now, we assign later

        # Validation logic
        if top_header_flag and (len(rec.get("numbers") or []) < 6 or not rec.get("strong")):
            return {"type": "header_skip", "y_span": (y0, y1)}

        ok = (len(rec.get("numbers") or []) >= 6) and (rec.get("strong") is not None)

        if not ok:
            w_band = crop_gray.shape[1]
            x0 = locate_vertical_separator(crop_gray)
            lpad = max(2, int(0.015 * w_band))
            shifts = [int(0.02 * w_band), int(0.035 * w_band)]
            salvaged = False
            for sh in shifts:
                xs = max(8, x0 - sh)
                left2 = crop_gray[:, :max(0, xs - lpad)]
                right2 = crop_gray[:, min(w_band, xs + lpad):]
                nums2, strong2, warn2 = perfected_ocr(left2, right2)
                if (len(nums2 or []) >= 6) and (strong2 is not None) and _safe_invariants_ok(nums2):
                    rec["numbers"], rec["strong"], rec["warnings"] = nums2, strong2, warn2
                    ok = True
                    salvaged = True
                    break
            if not salvaged:
                return None

        rec["y_abs"] = [int(y0), int(y1)]
        rec["row_height"] = int(y1 - y0)
        rec["median_row_height"] = med_row_height_all if med_row_height_all else None
        rec["fat_split"] = any(s <= y0 < e and y0 != s for (s, e, _) in events)
        return {"type": "result", "record": rec}

    # --- EXECUTE PARALLEL PROCESSING ---
    # Process all candidates found.
    work_items = [(j, span[0], span[1]) for j, span in enumerate(cand_spans)]

    valid_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_process_single_span, item): item for item in work_items}
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                if res["type"] == "header_skip":
                    header_skipped.append(res["y_span"])
                elif res["type"] == "result":
                    valid_results.append(res["record"])

    # Sort results top-to-bottom by their Y coordinate
    valid_results.sort(key=lambda r: r["y_abs"][0])
    
    # Assign sequential line indices
    results = []
    current_idx = start_index
    for rec in valid_results:
        rec["line_index"] = current_idx
        results.append(rec)
        current_idx += 1
        if len(results) >= 14: # Safety cap
            break

    # Fallback logic if nothing found
    if not results and bands1:
        se = bands1[0]
        y0, y1 = se
        crop_gray, _ = clip_with_min_pad(gray, y0, y1, pad=2)
        if crop_gray.size > 0:
            rec = split_row_and_ocr(crop_gray, start_index)
            if (len(rec.get("numbers") or []) >= 6) and (rec.get("strong") is not None):
                rec["y_abs"] = [int(y0), int(y1)]
                results.append(rec)

    # Post-process header dropping check on the final sorted list
    if results:
        y0, y1 = results[0].get("y_abs", [None, None])
        if y0 is not None and y1 is not None:
            band = gray[int(y0):int(y1)]
            if _should_drop_header(band) and (len(results[0].get("numbers") or []) < 6 or not results[0].get("strong")):
                header_skipped.append((int(y0), int(y1)))
                results = results[1:]

    return results