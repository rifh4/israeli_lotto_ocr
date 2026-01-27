LOCAL LOTTO OCR
===============

This project is a high-performance Optical Character Recognition (OCR) tool designed to extract lottery numbers from ticket images. It uses a hybrid approach combining Tesseract OCR, Computer Vision (OpenCV), and AI Verification (OpenAI) to ensure 100% accuracy.


FEATURES
--------
* Parallel Processing: Scans multiple rows simultaneously for high speed.
* Hybrid Engine: Uses classical CV for speed and AI for verifying ambiguous digits.
* Auto-Correction: Automatically corrects common OCR errors (e.g., confusing 7 and 1).
* Smart Parsing: Handles ticket headers, fractions (e.g., 6/37), and strong numbers.


PREREQUISITES
-------------
Before running the code, ensure you have the following installed:

1. Python 3.8+
   Make sure Python is installed and added to your system PATH.
   Download: https://www.python.org/downloads/

2. Tesseract OCR
   This project relies on the Tesseract engine. You must install it separately.

   * Windows: Download the installer (e.g., tesseract-ocr-w64-setup-v5.x.x.exe) from:
     https://github.com/UB-Mannheim/tesseract/wiki
   
   * Important: During installation, note the installation path (usually C:\Program Files\Tesseract-OCR).

   Configuration Option A (Recommended):
   Add Tesseract to your System PATH so it can be called from anywhere.

   Configuration Option B (Portable):
   Copy the installed Tesseract-OCR folder directly into this project's directory. The script is pre-configured to look for Tesseract-OCR/tesseract.exe inside the project folder.


INSTALLATION
------------
1. Unzip/Clone the project to a folder on your computer.

2. Open a terminal (Command Prompt or PowerShell) inside that folder.

3. Install Python dependencies:
   Run the following command to install numpy, opencv-python, pytesseract, and requests.
   
   pip install -r requirements.txt


AI CONFIGURATION (Optional but Recommended)
-------------------------------------------
For maximum accuracy, this tool uses the OpenAI API to verify "unsure" digits.

1. Get an API Key from https://platform.openai.com/
2. Set the Environment Variable:

   * Windows (PowerShell):
     $env:OPENAI_API_KEY="sk-your-api-key-here"

   * Windows (Command Prompt):
     set OPENAI_API_KEY=sk-your-api-key-here

   * Mac/Linux:
     export OPENAI_API_KEY="sk-your-api-key-here"

(Note: If you do not provide an API key, the script will run in "Local Only" mode using Tesseract and OpenCV templates.)


HOW TO RUN
----------
1. Start the script:
   python main.py

2. Enter the Image Path:
   The script will ask for the location of your image. You can drag and drop the file into the terminal window or paste the path.
   
   Example input: C:\Users\rifh\Desktop\ticket.jpg

3. View Results:
   The script will process the image (showing a spinner) and output the extracted numbers in JSON format.


PROJECT STRUCTURE
-----------------
* main.py - The entry point. Handles user input and displays results.
* lotto_extractor.py - Core logic for line extraction and parallel processing.
* image_utils.py - Image preprocessing (grayscale, deskew, loading).
* layout_analysis.py - Algorithms to find rows and digit boxes.
* ocr_engine.py - Handles Template Matching and Tesseract calls.
* ai_client.py - Communicates with OpenAI to verify ambiguous numbers.
* config.py - Central configuration for paths and constants.
* Digits/ - Folder containing reference digit images (0-9) for template matching.