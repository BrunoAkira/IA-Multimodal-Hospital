"""
Exemplo: características de voz de um arquivo WAV.

    python examples/voice_features_example.py caminho/do/audio.wav
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.io import wavfile

from src.audio.features import extract_features

if len(sys.argv) < 2:
    raise SystemExit("Uso: python examples/voice_features_example.py caminho/do/audio.wav")

sr, y = wavfile.read(sys.argv[1])
if y.ndim > 1:                                   # estéreo -> mono
    y = y.mean(axis=1)
if np.issubdtype(y.dtype, np.integer):           # inteiros -> faixa de -1 a 1
    y = y / np.iinfo(y.dtype).max
features = extract_features(y.astype(float), sr)
for name in ("speech_rate_sps", "pause_ratio", "long_pauses", "f0_mean_hz", "jitter", "shimmer", "hnr_db"):
    print(f"{name:18} {features[name]:.3f}")
