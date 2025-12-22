import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from inference import get_model
import supervision as sv
import cv2

API_KEY = "gkjsRK1Y61Jj2Ds9LuXj"
MODEL_ID = "drone-signal-detect-few-shot-7y5n8/1"

FS = 50_000_000
DURATION = 0.1
NFFT = 1024
NOVERLAP = 512
CMAP = "turbo"

def iq_to_spectrogram(iq_path):

    iq = np.fromfile(iq_path, dtype=np.complex64)
    total_samples = len(iq)
    samples_needed = int(FS * DURATION)

    if total_samples < samples_needed:
        samples_needed = total_samples

    iq = iq[:samples_needed]

    # Compute spectrogram
    f, t, Sxx = spectrogram(
        iq,
        fs=FS,
        nperseg=NFFT,
        noverlap=NOVERLAP,
        return_onesided=False,
        window='hann'
    )

    half = Sxx.shape[0] // 2
    Sxx = np.concatenate((Sxx[half:], Sxx[:half]), axis=0)
    f = np.concatenate((f[half:], f[:half]))

    Sxx_log = 10 * np.log10(Sxx + 1e-12)

    base = os.path.splitext(os.path.basename(iq_path))[0]
    out = f"{base}_spectrogram.png"

    fig, ax = plt.subplots(figsize=(14, 8))
    extent = [t.min()*1000, t.max()*1000, f.min()/1e6, f.max()/1e6]

    im = ax.imshow(Sxx_log, aspect='auto', origin='lower',
                   extent=extent, cmap=CMAP, interpolation='bilinear')

    ax.set_xlabel("Time (ms)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Frequency (MHz)", fontsize=14, fontweight="bold")
    ax.set_title("Drone RF Signal Spectrogram", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Power (dB)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return out


def run_rfdetr_detect(image_path):

    model = get_model(MODEL_ID, api_key=API_KEY)
    img = cv2.imread(image_path)

    results = model.infer(img)[0]
    detections = sv.Detections.from_inference(results)

    annotated = sv.BoxAnnotator().annotate(img, detections)
    annotated = sv.LabelAnnotator().annotate(annotated, detections)

    out_png = image_path.replace(".png", "_DETECTED.png")
    cv2.imwrite(out_png, annotated)

    return out_png
