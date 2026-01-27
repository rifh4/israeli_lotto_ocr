import sys
import json
import time
import threading
import itertools
import numpy as np

# Imports from our new modules
from image_utils import load_image_local_or_url, ensure_gray, deskew_soft
from ocr_engine import _load_digit_templates_once
from lotto_extractor import extract_all_lines, extract_single_line

# --- SPINNER UTILITY ---
class Spinner:
    def __init__(self, message="Processing..."):
        self.message = message
        self.stop_running = False
        self.thread = threading.Thread(target=self._animate)

    def _animate(self):
        for c in itertools.cycle(['|', '/', '-', '\\']):
            if self.stop_running:
                break
            # Overwrite the current line with the message and spinner char
            sys.stdout.write(f'\r{self.message} {c}')
            sys.stdout.flush()
            time.sleep(0.1)
        # Clear the line when done
        sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')
        sys.stdout.flush()

    def __enter__(self):
        self.stop_running = False
        self.thread.start()

    def __exit__(self, exc_type, exc_value, tb):
        self.stop_running = True
        self.thread.join()

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    
    # Pre-load resources
    _load_digit_templates_once()
    
    print("--- Local Lotto OCR ---")
    user_input = input("Enter Image URL or Local Path: ").strip().strip('"')
    
    if not user_input:
        print("Error: No input provided.")
        sys.exit(1)

    try:
        print(f"Loading: {user_input}...")
        
        # Load and preprocess image
        img = load_image_local_or_url(user_input)
        gray = ensure_gray(img)
        gray, angle = deskew_soft(gray)
        print(f"Image loaded (Skew correction: {angle:.2f} degrees)")

        lines = []
        
        # Attempt multi-line extraction with a spinner
        with Spinner("Scanning for multiple lines (this may take a moment)..."):
            lines = extract_all_lines(gray, start_index=1)
        
        # Fallback to single line if multi-line failed
        if not lines:
            print("Multi-line extraction found nothing, trying single line mode...")
            with Spinner("Scanning single line..."):
                lines = extract_single_line(gray, start_index=1)

        # Format output
        out_lines = []
        for rec in lines or []:
            nums = rec.get("numbers") or []
            strong = rec.get("strong")
            # Ensure we have exactly 6 numbers and a valid strong number before outputting
            if isinstance(strong, (int, np.integer)) and len(nums) == 6:
                out_lines.append({
                    "numbers": [f"{int(n):02d}" for n in nums],
                    "strong": int(strong)
                })

        print("\n--- JSON RESPONSE ---")
        print(json.dumps({"lines": out_lines}, indent=4))

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        sys.exit(1)