import soundfile as sf
import numpy as np

def wav_to_iq(wav_path, iq_path, scale_pcm=True):
    # Load WAV
    data, sr = sf.read(wav_path)   # data shape: (N, 2)
    
    if len(data.shape) != 2 or data.shape[1] != 2:
        raise ValueError("WAV file must have 2 channels (I = left, Q = right).")
    
    I = data[:, 0]
    Q = data[:, 1]

    # If PCM WAV, normalize (optional)
    if scale_pcm:
        if data.dtype == np.int16:
            I = I / 32768.0
            Q = Q / 32768.0
        elif data.dtype == np.int32:
            I = I / 2147483648.0
            Q = Q / 2147483648.0

    # Create complex IQ
    iq = I.astype(np.float32) + 1j * Q.astype(np.float32)

    # Save as raw .iq binary (complex64)
    iq.tofile(iq_path)

    print(f"Converted WAV → IQ")
    print(f"Samples: {len(iq)}")
    print(f"Sample rate (from WAV): {sr} Hz")
    print(f"Output saved: {iq_path}")

# Example
wav_to_iq("adsb.2021-11-26T15_03_30_573.wav", "output1.iq")
