import os
import pytesseract

# --- TESSERACT CONFIGURATION ---
# Get the absolute path to the folder where this script is running
script_dir = os.path.dirname(os.path.abspath(__file__))

# Point to the local tesseract.exe inside the project folder
local_tesseract = os.path.join(script_dir, "Tesseract-OCR", "tesseract.exe")

if os.path.exists(local_tesseract):
    pytesseract.pytesseract.tesseract_cmd = local_tesseract
else:
    # Fallback if the local folder isn't found
    cmd = os.getenv("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd

# Ensure Tesseract is actually working immediately on import
try:
    pytesseract.get_tesseract_version()
except Exception as e:
    # You can uncomment the line below if you want the program to strictly crash when Tesseract is missing
    # raise RuntimeError("Tesseract not available. Set TESSERACT_CMD or ensure it is on PATH.") from e
    print(f"Warning: Tesseract OCR not found or not executable. Error: {e}")

# --- HELPER FUNCTIONS ---
def is_truthy(x) -> bool:
    if x is None:
        return False
    return str(x).strip().lower() in ("1", "true", "yes", "y", "on")

# --- LOTTERY RULES ---
NUM_MIN, NUM_MAX = 1, 37
STR_MIN, STR_MAX = 1, 7

# --- TEMPLATE MATCHING THRESHOLDS ---
TEMPLATE_STRONG     = float(os.getenv("TEMPLATE_STRONG", "0.80"))
TEMPLATE_OK         = float(os.getenv("TEMPLATE_OK", "0.72"))
NUM_TEMPLATE_STRONG = float(os.getenv("NUM_TEMPLATE_STRONG", "0.74"))
NUM_TEMPLATE_OK     = float(os.getenv("NUM_TEMPLATE_OK", "0.66"))

# --- AI & API CONFIGURATION ---
# Centralized here to avoid repeated os.getenv calls in other files
ENABLE_AI = is_truthy(os.getenv("ENABLE_AI", "1"))
AI_MODEL_READ = os.getenv("OPENAI_MODEL_READ", None)
AI_TIMEOUT_MS = int(os.getenv("OPENAI_TIMEOUT_MS", "60000"))