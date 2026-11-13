"""
Gera dados sintéticos nos mesmos formatos das bases reais, para testar o pipeline completo:
  * base clínica: hosp/prescriptions, hosp/transfers, icu/chartevents (.csv.gz)
  * base de fala: Parquet com coluna de áudio e metadados
  * captura de movimento: arquivos .amc (S01/Marker/Clean/...)
Os valores são simulados e não têm significado clínico.
"""
from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "sample_datasets"
rng = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# base clínica
# ---------------------------------------------------------------------------
def make_clinical(n_patients: int = 12):
    hosp, icu = OUT / "clinical" / "hosp", OUT / "clinical" / "icu"
    hosp.mkdir(parents=True, exist_ok=True)
    icu.mkdir(parents=True, exist_ok=True)
    charts, rx, tr = [], [], []
    drugs = [("metoprolol", "PO", 25), ("heparin", "SC", 5000), ("insulin", "SC", 4),
             ("furosemide", "IV", 20), ("acetaminophen", "PO", 650), ("morphine", "IV", 2)]
    for p in range(n_patients):
        sid, hadm, stay = 10000000 + p, 20000000 + p, 30000000 + p
        t0 = pd.Timestamp("2150-01-01") + pd.Timedelta(days=int(rng.integers(0, 300)))
        hours = int(rng.integers(48, 120))
        hr = 80 + rng.normal(0, 4, hours).cumsum() * 0.2
        sbp = 120 + rng.normal(0, 5, hours).cumsum() * 0.2
        spo2 = np.clip(97 + rng.normal(0, 0.8, hours), 88, 100)
        rr = 16 + rng.normal(0, 1.5, hours)
        temp = 37 + rng.normal(0, 0.2, hours)
        if p % 3 == 0:  # alguns pacientes pioram no fim (sepse simulada)
            k = hours - 12
            hr[k:] += np.linspace(0, 45, hours - k)
            sbp[k:] -= np.linspace(0, 40, hours - k)
            spo2[k:] -= np.linspace(0, 8, hours - k)
            rr[k:] += np.linspace(0, 12, hours - k)
        for h in range(hours):
            t = t0 + pd.Timedelta(hours=h, minutes=int(rng.integers(0, 50)))
            for itemid, val in ((220045, hr[h]), (220179, sbp[h]), (220180, sbp[h] * 0.6),
                                (220277, spo2[h]), (220210, rr[h]), (223761, temp[h] * 9 / 5 + 32)):
                charts.append((sid, hadm, stay, t, itemid, round(float(val), 1)))
        # prescrições: alguns remédios com reajustes
        for drug, route, dose in drugs[: 3 + p % 4]:
            t = t0 + pd.Timedelta(hours=int(rng.integers(0, 12)))
            for k in range(int(rng.integers(1, 5))):
                d = dose * (2 if (p % 4 == 1 and k == 2) else 1)  # salto de dose
                r = "IV" if (p % 5 == 2 and k == 1) else route    # troca de via
                rx.append((sid, hadm, t, t + pd.Timedelta(hours=24), drug, str(d), "mg", r, 2))
                t += pd.Timedelta(hours=int(rng.integers(6, 30)))
        # transferências: ED -> Med -> MICU (-> Med -> MICU em alguns)
        units = ["Emergency Department", "Medicine", "Medical Intensive Care Unit (MICU)"]
        if p % 4 == 0:
            units += ["Medicine", "Medical Intensive Care Unit (MICU)"]
        t = t0 - pd.Timedelta(hours=6)
        for i, u in enumerate(units):
            dur = pd.Timedelta(hours=int(rng.integers(5, 30)))
            tr.append((sid, hadm, 40000000 + p * 10 + i, "admit" if i == 0 else "transfer", u, t, t + dur))
            t += dur
        tr.append((sid, hadm, 40000000 + p * 10 + 9, "discharge", None, t, None))

    pd.DataFrame(charts, columns=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"]) \
      .assign(value=lambda d: d["valuenum"].astype(str)).to_csv(icu / "chartevents.csv.gz", index=False)
    pd.DataFrame(rx, columns=["subject_id", "hadm_id", "starttime", "stoptime", "drug", "dose_val_rx",
                              "dose_unit_rx", "route", "doses_per_24_hrs"]).to_csv(hosp / "prescriptions.csv.gz", index=False)
    pd.DataFrame(tr, columns=["subject_id", "hadm_id", "transfer_id", "eventtype", "careunit", "intime",
                              "outtime"]).to_csv(hosp / "transfers.csv.gz", index=False)
    print("base clínica de amostra ->", OUT / "clinical")


# ---------------------------------------------------------------------------
# Áudio (sinal "parecido com fala": vogais harmônicas separadas por pausas)
# ---------------------------------------------------------------------------
def synth_utterance(dysarthric: bool, tired: bool = False, sr: int = 16000) -> np.ndarray:
    n_syll = int(rng.integers(4, 9))
    f0 = rng.uniform(100, 220)
    parts = []
    for _ in range(n_syll):
        dur = rng.uniform(0.25, 0.45) if dysarthric else rng.uniform(0.12, 0.25)
        t = np.arange(int(dur * sr)) / sr
        jitter = 0.04 if dysarthric else 0.008
        f = f0 * (1 + rng.normal(0, jitter, len(t)).cumsum() / len(t) * 5)
        phase = 2 * np.pi * np.cumsum(f) / sr
        s = sum(np.sin(k * phase) / k for k in range(1, 6))
        s *= np.hanning(len(t)) * (0.3 if tired else 0.6)
        s += rng.normal(0, 0.06 if dysarthric else 0.015, len(t))   # ruído = pior HNR
        parts.append(s)
        pause = rng.uniform(0.15, 0.6) if (dysarthric or tired) else rng.uniform(0.03, 0.12)
        parts.append(rng.normal(0, 0.003, int(pause * sr)))
    return np.concatenate(parts).astype(np.float32) * 0.8


def wav_bytes(y: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def make_speech(per_speaker: int = 25):
    speakers = {"F01": "dysarthria", "M01": "dysarthria", "M03": "dysarthria",
                "FC01": "healthy", "MC01": "healthy", "MC02": "healthy"}
    words = ["yes", "no", "the quick brown fox", "pat", "window", "say ah", "she sells seashells"]
    rows = []
    for spk, status in speakers.items():
        for i in range(per_speaker):
            tired = status == "healthy" and i >= per_speaker - 4   # últimas falas "cansadas"
            y = synth_utterance(status == "dysarthria", tired)
            rows.append({"audio": {"bytes": wav_bytes(y), "path": f"{spk}_{i:03d}.wav"},
                         "transcription": str(rng.choice(words)), "speech_status": status,
                         "gender": "female" if spk.startswith("F") else "male", "speaker_id": spk})
    out = OUT / "speech"
    out.mkdir(parents=True, exist_ok=True)
    try:
        pd.DataFrame(rows).to_parquet(out / "train-00000-of-00004.parquet", index=False)
        print("base de fala de amostra ->", out)
    except ImportError:
        print("pyarrow não instalado: Parquet de amostra da base de fala não foi gerado.")
    return rows


# ---------------------------------------------------------------------------
# captura de movimento (.amc: ângulos das articulações por quadro)
# ---------------------------------------------------------------------------
def amc_text(frames: list[dict]) -> str:
    lines = [":FULLY-SPECIFIED", ":DEGREES"]
    for i, f in enumerate(frames, start=1):
        lines.append(str(i))
        for joint, vals in f.items():
            lines.append(joint + " " + " ".join(f"{v:.4f}" for v in vals))
    return "\n".join(lines) + "\n"


def squat_frames(paces=(1.0,), depth: float = 80, fps: int = 30, bad_form: bool = False) -> list[dict]:
    frames = []
    for pace in paces:   # pace 1 = normal, <1 = rápido, >1 = lento
        n = int(3.0 * pace * fps)
        phase = np.sin(np.linspace(0, np.pi, n)) ** 2
        for k in range(n):
            knee = depth * phase[k] + rng.normal(0, 1)
            asym = 25 * phase[k] if bad_form else 0     # um lado dobra menos
            frames.append({
                "root": [0, 900 - 3 * knee, 0, 0, 0, 0],
                "lfemur": [-knee * 0.8, 0, 2], "rfemur": [-(knee - asym) * 0.8, 0, -2],
                "ltibia": [knee], "rtibia": [knee - asym],
                "lhumerus": [-60 * phase[k], 0, 0], "rhumerus": [-60 * phase[k], 0, 0],
                "lradius": [10], "rradius": [10],
            })
        frames.extend([frames[-1]] * 10)   # pequena pausa entre repetições
    return frames


def make_motion_capture(subjects: int = 4):
    for s in range(1, subjects + 1):
        sid = f"S{s:02d}"
        d = OUT / "motion_capture" / sid / "Marker" / "Clean"
        d.mkdir(parents=True, exist_ok=True)
        for motion, depth in (("MiniSquat", 70), ("SitToStand", 90)):
            bad = (s == subjects and motion == "MiniSquat")   # último sujeito com execução ruim
            depth_s = depth * rng.uniform(0.9, 1.1)
            (d / f"{motion}R1{sid}.amc").write_text(amc_text(squat_frames((1.0,), depth_s)))
            (d / f"{motion}R3{sid}.amc").write_text(
                amc_text(squat_frames((1.0, 0.55, 1.8), depth_s, bad_form=bad)))
    print("captura de movimento de amostra ->", OUT / "motion_capture")


def copy_consultations():
    """Copia os roteiros de consulta do projeto, para a pasta sintética ter a mesma estrutura de data/."""
    import shutil
    src = Path(__file__).resolve().parents[1] / "data" / "consultations"
    dst = OUT / "consultations"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("consult_*.txt"):
        shutil.copy(f, dst / f.name)
    print("consultas ->", dst)


if __name__ == "__main__":
    make_clinical()
    make_speech()
    make_motion_capture()
    copy_consultations()
