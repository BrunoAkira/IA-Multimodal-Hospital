"""
Extração de características acústicas da voz (só numpy/scipy: leve e sem dependências extras).

O que cada característica indica (literatura de fala patológica):
  * speech_rate / pause_ratio / long_pauses -> fala lenta e pausada (disartria, cansaço,
    pausas para respirar = dificuldade respiratória)
  * f0_mean / f0_std_st                    -> tom de voz e sua variação (voz monótona = cansaço)
  * jitter / shimmer                       -> instabilidade da frequência/amplitude (disartria)
  * hnr_db                                  -> relação harmônico-ruído (voz soprosa/rouca, baixa = pior)
  * rms_db / rms_cv                         -> volume e sua estabilidade (voz fraca = cansaço)
  * mfcc_1..13                              -> "timbre" geral (usado pelo classificador)

Obs.: jitter/shimmer aqui são aproximações por quadro (frame), não o cálculo
ciclo a ciclo do Praat. Para o objetivo do projeto (comparar falantes) isso basta.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dct
from scipy.signal import find_peaks

FRAME_S = 0.04   # 40 ms por quadro (cabe 2+ períodos de voz grave)
HOP_S = 0.01     # avança 10 ms por quadro


def _frames(y: np.ndarray, sr: int) -> np.ndarray:
    """Divide o sinal em quadros sobrepostos (matriz n_quadros x tamanho_quadro)."""
    win, hop = int(FRAME_S * sr), int(HOP_S * sr)
    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))
    n = 1 + (len(y) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return y[idx]


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 26) -> np.ndarray:
    """Filtros triangulares na escala Mel (como o ouvido humano percebe frequências)."""
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = inv(np.linspace(mel(0), mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        a, b, c = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, a:b] = (np.arange(a, b) - a) / max(b - a, 1)
        fb[m - 1, b:c] = (c - np.arange(b, c)) / max(c - b, 1)
    return fb


def extract_features(y: np.ndarray, sr: int) -> dict:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y - y.mean()
    peak = np.abs(y).max()
    if peak < 1e-6:
        raise ValueError("Áudio vazio/silencioso")
    y = y / peak                                   # normaliza volume entre gravações

    F = _frames(y, sr) * np.hanning(int(FRAME_S * sr))
    rms = np.sqrt((F ** 2).mean(axis=1)) + 1e-10
    rms_db = 20 * np.log10(rms)

    # --- fala x silêncio: quadro com energia > 15% do percentil 95 = fala ---
    voiced_energy = rms > 0.15 * np.percentile(rms, 95)
    speech_idx = np.flatnonzero(voiced_energy)
    first, last = speech_idx[0], speech_idx[-1]
    span_s = (last - first + 1) * HOP_S

    # pausas = trechos silenciosos ENTRE o primeiro e o último quadro com fala
    silent = ~voiced_energy[first:last + 1]
    runs, cur = [], 0
    for s in silent:
        if s:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    pauses_s = np.array(runs) * HOP_S
    pauses_s = pauses_s[pauses_s >= 0.15]          # menos de 150 ms não é pausa perceptível

    # --- sílabas ~ picos de energia (proxy de velocidade de fala) ---
    smooth = np.convolve(rms[first:last + 1], np.ones(5) / 5, mode="same")
    peaks, _ = find_peaks(smooth, prominence=0.1 * smooth.max(), distance=int(0.1 / HOP_S))

    # --- F0 e HNR por autocorrelação (calculada via FFT para ser rápida) ---
    Fv = F[voiced_energy]
    n_fft = 2 * Fv.shape[1]
    spec = np.fft.rfft(Fv, n=n_fft)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:, : Fv.shape[1]]
    ac = ac / (ac[:, :1] + 1e-12)                  # normaliza: r(0) = 1
    lo, hi = int(sr / 400), int(sr / 70)           # voz humana entre 70 e 400 Hz
    lags = lo + np.argmax(ac[:, lo:hi], axis=1)
    r = ac[np.arange(len(lags)), lags]
    pitched = r > 0.45                             # periodicidade suficiente = vozeado
    f0 = sr / lags[pitched]
    r_p = np.clip(r[pitched], 1e-3, 0.999)
    hnr = 10 * np.log10(r_p / (1 - r_p))           # fórmula clássica de Boersma

    periods = 1 / f0 if len(f0) else np.array([np.nan])
    amps = np.abs(Fv[pitched]).max(axis=1) if pitched.any() else np.array([np.nan])
    jitter = np.mean(np.abs(np.diff(periods))) / np.mean(periods) if len(periods) > 2 else np.nan
    shimmer = np.mean(np.abs(np.diff(amps))) / np.mean(amps) if len(amps) > 2 else np.nan

    # --- espectro: centroide e MFCC ---
    power = np.abs(np.fft.rfft(F, n=512)) ** 2
    freqs = np.fft.rfftfreq(512, 1 / sr)
    centroid = (power[voiced_energy] @ freqs) / (power[voiced_energy].sum(axis=1) + 1e-12)
    mel = np.log(power[voiced_energy] @ _mel_filterbank(sr, 512).T + 1e-10)
    mfcc = dct(mel, type=2, axis=1, norm="ortho")[:, :13].mean(axis=0)

    feats = {
        "duration_s": len(y) / sr,
        "speech_span_s": span_s,
        "speech_rate_sps": len(peaks) / span_s if span_s > 0 else 0.0,   # sílabas por segundo
        "pause_ratio": pauses_s.sum() / span_s if span_s > 0 else 0.0,
        "n_pauses": len(pauses_s),
        "long_pauses": int((pauses_s >= 0.4).sum()),                      # pausas "de respiração"
        "voiced_fraction": float(pitched.mean()) if len(pitched) else 0.0,
        "rms_db": float(rms_db[voiced_energy].mean()),
        "rms_cv": float(rms[voiced_energy].std() / rms[voiced_energy].mean()),
        "f0_mean_hz": float(np.median(f0)) if len(f0) else np.nan,
        "f0_std_st": float(np.std(12 * np.log2(f0 / np.median(f0)))) if len(f0) > 2 else np.nan,
        "jitter": float(jitter),
        "shimmer": float(shimmer),
        "hnr_db": float(np.median(hnr)) if len(hnr) else np.nan,
        "spectral_centroid_hz": float(np.mean(centroid)),
    }
    feats.update({f"mfcc_{i + 1}": float(v) for i, v in enumerate(mfcc)})
    return feats
