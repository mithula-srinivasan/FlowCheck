import librosa
import numpy as np

def extract_audio_features(audio_path):
    """
    Extract 5 core features from audio:
      1. MFCC mean (spectral representation)
      2. Zero Crossing Rate (ZCR)
      3. RMS energy
      4. Pitch standard deviation
      5. Pause ratio
    """
    try:
        y, sr = librosa.load(audio_path, sr=16000)

        # --- MFCC Mean ---
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs)

        # --- Zero Crossing Rate ---
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        # --- RMS Energy ---
        rms = np.mean(librosa.feature.rms(y=y))

        # --- Pitch Variability ---
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0

        # --- Pause Ratio ---
        pauses = np.sum(np.abs(y) < 0.01)
        pause_ratio = pauses / len(y)

        return [mfcc_mean, zcr, rms, pitch_std, pause_ratio]

    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None
