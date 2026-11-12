"""
Gera a pasta example_data/ (versionada no repositório) a partir das bases completas em datasets/.

Conteúdo gerado, pequeno o bastante para o Git:
  * clinical/        tabelas da base clínica: chartevents só com os sinais vitais usados,
                     prescriptions e transfers completos
  * motion_capture/  alguns exercícios de um participante (arquivos .amc R1 e R3, sistema "marker")
  * exercise_videos/ gabarito (Segmentation.csv) só dos vídeos escolhidos + ângulos já calculados
                     (angles/*.csv), que permitem rodar a validação sem os vídeos completos
  * physio_videos/   dois trechos curtos de vídeo (640 px, 15 quadros/s) para a análise com
                     OpenPose/YOLOv8
  * README.md        origem e licença de cada amostra

A base de fala não é redistribuída (licença de uso acadêmico); ela é baixada por
scripts/download_example_data.py.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

VIDEO_EXERCISES = {1: "arm_raise", 6: "squat"}   # exercise_id -> prefixo do trecho de vídeo


def size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def build_clinical(src: Path, out: Path):
    """chartevents filtrado para os itemids de sinais vitais; demais tabelas copiadas."""
    def find(name):
        hits = sorted(src.rglob(f"{name}.csv.gz")) or sorted(src.rglob(f"{name}.csv"))
        if not hits:
            raise FileNotFoundError(f"{name} não encontrado em {src}")
        return hits[0]

    (out / "hosp").mkdir(parents=True, exist_ok=True)
    (out / "icu").mkdir(parents=True, exist_ok=True)
    itemids = {i for ids in config.VITAL_ITEMIDS.values() for i in ids}
    parts = []
    for chunk in pd.read_csv(find("chartevents"), chunksize=config.CSV_CHUNK_ROWS, low_memory=False):
        parts.append(chunk[chunk["itemid"].isin(itemids)])
    pd.concat(parts).to_csv(out / "icu" / "chartevents.csv.gz", index=False)
    for name in ("prescriptions", "transfers"):
        shutil.copy(find(name), out / "hosp" / f"{name}.csv.gz")
    print(f"  clínica: {size_mb(out):.1f} MB")


def build_motion_capture(src: Path, out: Path, n_motions: int):
    """Para o primeiro participante: n movimentos, arquivos R1 e R3 do sistema 'marker'."""
    from src.video.amc import index_motion_capture
    idx = index_motion_capture(src)
    idx = idx[(idx["system"] == "marker") & idx["reps"].isin([1, 3])]
    subject = sorted(idx["subject"].unique())[0]
    sub = idx[idx["subject"] == subject]
    motions = [m for m, g in sub.groupby("motion") if set(g["reps"]) >= {1, 3}][:n_motions]
    for _, r in sub[sub["motion"].isin(motions)].iterrows():
        f = Path(r["path"])
        dest = out / f.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dest)
    print(f"  captura de movimento: {subject}, {len(motions)} movimentos, {size_mb(out):.1f} MB")


def trim_clip(video: Path, dest: Path, start_s: float, seconds: float, width: int = 640, fps: float = 15):
    """Recorta um trecho do vídeo, redimensionado, usando só o OpenCV."""
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(src_fps / fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * src_fps))
    writer, written, i = None, 0, 0
    while written < seconds * fps:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            h = int(frame.shape[0] * width / frame.shape[1])
            frame = cv2.resize(frame, (width, h))
            if writer is None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, h))
            writer.write(frame)
            written += 1
        i += 1
    cap.release()
    if writer is not None:
        writer.release()


def build_exercise_videos(src: Path, out: Path, angles_dir: Path, per_exercise: int, clip_seconds: float):
    """Gabarito filtrado + ângulos já calculados + um trecho curto de vídeo por exercício."""
    seg_path = next(src.rglob("Segmentation.csv"))
    seg = pd.read_csv(seg_path, sep=";")
    (out / "angles").mkdir(parents=True, exist_ok=True)
    chosen = []
    for ex, prefix in VIDEO_EXERCISES.items():
        ids = sorted(set(seg.loc[seg["exercise_id"] == ex, "video_id"]))
        with_angles = [v for v in ids if (angles_dir / f"{v}_Camera17.csv").exists()][:per_exercise]
        if not with_angles:
            print(f"  aviso: exercício {ex} sem ângulos calculados em {angles_dir} "
                  f"(rode antes: python main.py video-validation)")
        for v in with_angles:
            shutil.copy(angles_dir / f"{v}_Camera17.csv", out / "angles" / f"{v}_Camera17.csv")
        chosen += with_angles
        # trecho de vídeo do primeiro vídeo do exercício, começando na primeira repetição
        videos = sorted(src.rglob(f"Ex{ex}/*Camera17*.mp4"))
        if videos:
            vid = videos[0]
            vid_id = vid.name.split("-")[0]
            first = seg.loc[seg["video_id"] == vid_id, "first_frame"]
            start = max(0.0, (first.min() if len(first) else 0) / 30 - 1)
            trim_clip(vid, out.parent / "physio_videos" / f"{prefix}_example.mp4", start, clip_seconds)
    seg[seg["video_id"].isin(chosen)].to_csv(out / "Segmentation.csv", sep=";", index=False)
    txt = next(src.rglob("Segmentation.txt"), None)
    if txt:
        shutil.copy(txt, out / "Segmentation.txt")
    print(f"  vídeos de exercícios: {len(chosen)} vídeos (ângulos + gabarito), {size_mb(out):.1f} MB")
    print(f"  trechos de vídeo: {size_mb(out.parent / 'physio_videos'):.1f} MB")


README = """# example_data/

Amostras pequenas de bases públicas, incluídas para que o projeto rode sem downloads grandes:

```bash
export HOSPITAL_AI_DATASETS=example_data
python scripts/download_example_data.py      # base de fala (não redistribuída aqui)
python main.py all
python main.py video-validation
```

| Pasta | Conteúdo | Origem | Licença |
|---|---|---|---|
| `clinical/` | Sinais vitais (só os itens usados), prescrições e transferências | MIMIC-IV Clinical Database Demo v2.2 (PhysioNet) | Open Data Commons ODbL v1.0 |
| `motion_capture/` | Alguns exercícios de um participante (`.amc`) | UL-RED — University of Liverpool Rehabilitation Exercise Dataset | CC BY 4.0 |
| `exercise_videos/` | Gabarito dos vídeos escolhidos e ângulos das articulações já calculados | REHAB24-6 (Černek, Sedmidubsky, Budikova, SISAP 2024) | CC BY-NC 4.0 |
| `physio_videos/` | Trechos curtos de vídeo (640 px, 15 quadros/s) | REHAB24-6 | CC BY-NC 4.0 |
| `speech/` | Baixado por `scripts/download_example_data.py` (não versionado) | TORGO Database (Rudzicz et al., 2012) | Uso acadêmico, conforme os autores |

As amostras são subconjuntos das bases originais, sem alterações nos valores. Referências completas
no README principal.
"""


def main():
    ap = argparse.ArgumentParser(description="Gera example_data/ a partir das bases completas")
    ap.add_argument("--source", default=str(ROOT / "datasets"))
    ap.add_argument("--out", default=str(ROOT / "example_data"))
    ap.add_argument("--angles", default=str(config.RESULTS_DIR / "video_validation_angles"))
    ap.add_argument("--motions", type=int, default=4, help="movimentos de captura incluídos")
    ap.add_argument("--videos", type=int, default=4, help="vídeos com gabarito por exercício")
    ap.add_argument("--clip-seconds", type=float, default=20)
    args = ap.parse_args()

    src, out = Path(args.source), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Gerando {out} a partir de {src}")
    build_clinical(src / "clinical", out / "clinical")
    build_motion_capture(src / "motion_capture", out / "motion_capture", args.motions)
    build_exercise_videos(src / "exercise_videos", out / "exercise_videos", Path(args.angles),
                          args.videos, args.clip_seconds)
    (out / "README.md").write_text(README, encoding="utf-8")
    print(f"Total: {size_mb(out):.1f} MB")


if __name__ == "__main__":
    main()
