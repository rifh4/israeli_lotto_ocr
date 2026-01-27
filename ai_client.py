import os
import json
import base64
import requests
import cv2
import numpy as np
from typing import Optional, List, Sequence, Dict, Tuple, Any

# Imports from our new modules
from config import (
    NUM_MIN, NUM_MAX,
    AI_MODEL_READ, AI_TIMEOUT_MS
)

# --- HELPER FUNCTIONS ---

def _get_api_key() -> str:
    """Retrieves API Key from environment variable."""
    return os.getenv("OPENAI_API_KEY", "").strip()

def _build_tens_candidates(v: int, lo: int, hi: int) -> List[int]:
    """Build candidate numbers by varying tens place while keeping ones."""
    c = set()
    if isinstance(v, int):
        c.add(v)
        if v >= 10:
            ones = v % 10
            for t in (1, 2, 3):
                x = 10 * t + ones
                if lo <= x <= hi:
                    c.add(x)
    out = [x for x in sorted(c) if lo <= x <= hi]
    return out

def _png_data_url(img: np.ndarray) -> str:
    """Encode an image as a PNG data URL for API call."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return "data:image/png;base64," + b64

# --- AI API FUNCTIONS ---

def _ai_verify_choice(slot_img: np.ndarray, candidates: Sequence[int], model: Optional[str] = None, timeout_ms: int = 60000) -> Optional[int]:
    """Ask ChatGPT via OpenAI API to pick one value from candidate list based on an image crop."""
    key = _get_api_key()
    if not key or not candidates:
        return None
        
    url = "https://api.openai.com/v1/chat/completions"
    # Use the passed model, or the config global, or fallback to default
    m = model or AI_MODEL_READ or "gpt-4o-mini"
    
    data_url = _png_data_url(slot_img)
    if not data_url:
        return None
        
    cand_str = ",".join(str(x) for x in candidates)
    sys_prompt = "You are an OCR verifier. Pick exactly one value from the provided candidate list."
    user_text = f"Candidates: [{cand_str}]. Return JSON {{\"choice\": <one of the candidates as integer>}}."
    
    msg = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": data_url}}
        ]}
    ]
    
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": m, 
        "messages": msg, 
        "temperature": 0, 
        "response_format": {"type": "json_object"}, 
        "max_tokens": 50
    }
    
    # Use config timeout if not overridden
    t_val = timeout_ms if timeout_ms is not None else AI_TIMEOUT_MS
    timeout_sec = max(0.1, float(t_val) / 1000.0)
    
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
        j = r.json()
        c = j.get("choices", [])
        if not c:
            return None
        content = c[0]["message"]["content"]
        obj = json.loads(content)
        val = obj.get("choice", None)
        if isinstance(val, str) and val.isdigit():
            val = int(val)
        if isinstance(val, int) and val in set(candidates):
            return val
        return None
    except Exception:
        return None

def ai_verify_numbers_in_left(left2: np.ndarray, classic_numbers: List[int], ambiguous_only: bool = True, model: Optional[str] = None, timeout_ms: int = 60000) -> Tuple[List[int], Dict]:
    """API-verify ambiguous slots on the left by constrained choice."""
    
    # Local helper to split image into n slots
    def _equal_bins_slots_local(img_arr, n=6):
        h_img, w_img = img_arr.shape[:2]
        gutter = max(4, int(0.01 * w_img))
        w_slot = w_img // n
        slots = []
        x = 0
        for i in range(n):
            x0 = max(0, x + gutter // 2)
            x1 = min(w_img, (x + w_slot) - gutter // 2)
            if i == 0: x0 = 0
            if i == n - 1: x1 = w_img
            slots.append((x0, x1))
            x += w_slot
        return slots
    
    H, W = left2.shape[:2]
    slots = _equal_bins_slots_local(left2, n=6)
    out = list(classic_numbers)[:6] + [None] * max(0, 6 - len(classic_numbers))
    info = {"slots": []}
    
    for i in range(min(6, len(slots))):
        v = out[i]
        if not isinstance(v, int):
            continue
        if ambiguous_only and v < 10:
            continue
            
        cand = _build_tens_candidates(v, NUM_MIN, NUM_MAX)
        if len(cand) <= 1:
            continue
            
        x0, x1 = slots[i]
        x0 = max(0, x0)
        x1 = min(W, x1)
        crop = left2[:, x0:x1]
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        
        choice = _ai_verify_choice(crop, cand, model=model, timeout_ms=timeout_ms)
        used = False
        if isinstance(choice, int) and choice in cand:
            if choice != v:
                out[i] = choice
            used = True
        info["slots"].append({
            "index": i + 1, 
            "classic": str(v), 
            "candidates": [str(x) for x in cand], 
            "ai_choice": (str(choice) if choice is not None else None), 
            "used": bool(used)
        })
    return out[:6], info

def ai_read_numbers_from_left(left2: np.ndarray, model: Optional[str] = None, timeout_ms: int = 60000) -> Tuple[Optional[List[int]], Dict]:
    """Ask OpenAI API to directly read six numbers from the left strip."""
    key = _get_api_key()
    if not key:
        return None, {}
        
    url = "https://api.openai.com/v1/chat/completions"
    m = model or AI_MODEL_READ or "gpt-4o-mini"
    
    img = cv2.resize(left2, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    data_url = _png_data_url(img)
    if not data_url:
        return None, {}
        
    sys_prompt = "You are an OCR model. Read exactly six numbers in left-to-right order as printed in the image."
    user_text = "Return JSON {\"numbers\":[n1,n2,n3,n4,n5,n6]} using integers only."
    
    msg = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": data_url}}
        ]}
    ]
    
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": m, 
        "messages": msg, 
        "temperature": 0, 
        "response_format": {"type": "json_object"}, 
        "max_tokens": 100
    }
    
    t_val = timeout_ms if timeout_ms is not None else AI_TIMEOUT_MS
    timeout_sec = max(0.1, float(t_val) / 1000.0)
    
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
        j = r.json()
        c = j.get("choices", [])
        if not c:
            return None, {"raw": j}
        content = c[0]["message"]["content"]
        obj = json.loads(content)
        arr = obj.get("numbers")
        if not isinstance(arr, list) or len(arr) != 6:
            return None, {"raw": obj}
        out = []
        for v in arr:
            if isinstance(v, str) and v.isdigit():
                v = int(v)
            if not isinstance(v, int):
                return None, {"raw": obj}
            out.append(v)
        return out, {"raw": obj}
    except Exception:
        return None, {}

# --- VALIDATION & MERGE LOGIC ---

def _safe_invariants_ok(nums: Sequence[Any]) -> bool:
    """Validate basic invariants for six unique in-range numbers."""
    try:
        vals = []
        for x in nums:
            if isinstance(x, (int, np.integer)):
                vals.append(int(x))
            elif isinstance(x, str) and x.isdigit():
                vals.append(int(x))
        if not vals:
            return False
        if len(set(vals)) != len(vals):
            return False
        for n in vals:
            if n < NUM_MIN or n > NUM_MAX:
                return False
        return True
    except Exception:
        return False

def ai_confirm_and_merge_with_read(classic_numbers: List[int], verify: Dict, ai_nums: Optional[List[int]], strong: Optional[int]) -> Tuple[List[int], Dict]:
    """Merge API outputs with classic read."""
    base = list(classic_numbers)[:6] + [None] * max(0, 6 - len(classic_numbers))
    out = base[:6]
    info = {"confirmed_slots": [], "rejected_slots": []}

    if not isinstance(ai_nums, list) or len(ai_nums) != 6:
        return out, info

    slots = verify.get("slots") if isinstance(verify, dict) else None
    if isinstance(slots, list):
        for s in slots:
            try:
                idx = int(s.get("index")) - 1
                used = bool(s.get("used"))
                choice = s.get("ai_choice")
                if isinstance(choice, str) and choice.isdigit():
                    choice = int(choice)
                if used and isinstance(choice, int) and 0 <= idx < 6 and ai_nums[idx] == choice:
                    trial = out[:]
                    trial[idx] = choice
                    if _safe_invariants_ok(trial):
                        out[idx] = choice
                        info["confirmed_slots"].append(idx + 1)
                    else:
                        info["rejected_slots"].append(idx + 1)
            except Exception:
                pass

    ai_valid = _safe_invariants_ok(ai_nums)
    if ai_valid:
        for i in range(6):
            vi = out[i] if i < len(out) else None
            if not isinstance(vi, int) or vi < NUM_MIN or vi > NUM_MAX:
                cand = ai_nums[i]
                if isinstance(cand, int) and NUM_MIN <= cand <= NUM_MAX and cand not in out:
                    out[i] = cand
                    info["confirmed_slots"].append(i + 1)

        if not _safe_invariants_ok(out):
            out = list(ai_nums)
            info["reordered_from_ai"] = True
        elif set(out) == set(ai_nums) and out != ai_nums:
            out = list(ai_nums)
            info["reordered_from_ai"] = True

    return out, info