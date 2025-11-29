from full_pipeline import iq_to_spectrogram, run_rfdetr_detect
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend folder
ROOT_DIR = os.path.dirname(BASE_DIR)                  # dashboard folder
OUTPUT_DIR = os.path.join(ROOT_DIR, "generated")

os.makedirs(OUTPUT_DIR, exist_ok=True)


OUTPUT_DIR = "../generated/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_iq_file(iq_path):

    # 1. Generate spectrogram using YOUR FUNCTION
    spec_file = iq_to_spectrogram(iq_path)

    # Copy to dashboard static image name
    spec_dashboard = os.path.join(OUTPUT_DIR, "spectrogram.png")
    shutil.copy(spec_file, spec_dashboard)

    # 2. Run RF-DETR using YOUR FUNCTION
    detected_file = run_rfdetr_detect(spec_file)

    detected_dashboard = os.path.join(OUTPUT_DIR, "detected.png")
    shutil.copy(detected_file, detected_dashboard)

    return {
        "spectrogram": "spectrogram.png",
        "detected": "detected.png",
        "status": "success"
    }
