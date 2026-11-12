"""Orquestra o módulo de vídeo: captura de movimento (.amc) + vídeos .mp4 (OpenPose/YOLOv8) + relatórios."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from .. import config
from ..core.events import save_events, clear_events
from ..core.report import write_report, fig_to_b64, file_to_b64
from . import motion


def _motion_capture_reports(df: pd.DataFrame) -> list[str]:
    """Um relatório HTML por sujeito com os desvios de cada repetição."""
    paths = []
    flag_cols = ["pace_deviation", "reduced_rom", "asymmetry", "form_deviation"]
    for subj, g in df.groupby("subject"):
        table = g[["motion", "rep", "pace_label", "duration_ratio", "rom_ratio", "dtw_shape",
                   "asymmetry_delta", "anomaly_score"] + flag_cols].copy()
        table["findings"] = g[flag_cols].apply(lambda r: ", ".join(c for c in flag_cols if r[c]) or "-", axis=1)
        table = table.drop(columns=flag_cols)

        images = []
        for mot, gm in g.groupby("motion"):
            fig, ax = plt.subplots(figsize=(8, 2.6))
            ax.plot(gm.iloc[0]["_ref_curve"], color="black", lw=2, label="reference (R1)")
            for _, r in gm.iterrows():   # iterrows: colunas com "_" no início não somem
                ax.plot(r["_rep_curve"], lw=1.2, label=f"rep {r['rep']} ({r['pace_label']})")
            ax.set_title(f"{mot} — {gm.iloc[0]['main_dof']} angle per frame", fontsize=10)
            ax.set_xlabel("frame"); ax.set_ylabel("degrees"); ax.legend(fontsize=7)
            images.append(fig_to_b64(fig))
        p = write_report(f"physio_{subj}.html", f"Physiotherapy session report — {subj}", [
            {"heading": "Repetition deviations vs. reference",
             "text": "duration_ratio > 1 = slower; rom_ratio < 1 = smaller range of motion; "
                     "dtw_shape = curve shape difference; anomaly_score 0..1.", "table": table},
            {"heading": "Movement curves", "images": images},
        ])
        paths.append(str(p))
    return paths


def run_video() -> dict:
    summary, events = {}, []

    # --- 1) captura de movimento: arquivos .amc ---
    try:
        reps = motion.build_repetition_table()
        scored, metrics = motion.score_repetitions(reps)
        scored.drop(columns=["_ref_curve", "_rep_curve"]).to_csv(config.RESULTS_DIR / "physio_repetitions.csv", index=False)
        events += motion.repetition_events(scored, config.RELATIVE_EPOCH)
        summary["motion_capture"] = metrics
        summary["motion_capture_reports"] = _motion_capture_reports(scored)
        print(f"[video] captura de movimento: {metrics}")
    except FileNotFoundError as exc:
        print(f"[video] {exc}")

    # --- 2) Vídeos .mp4 com OpenPose + YOLOv8 ---
    # vídeos locais (data/physio_videos) + vídeos da pasta de dados ativa (<dados>/physio_videos)
    folders = [config.PHYSIO_VIDEO_DIR, config.DATASETS / "physio_videos"]
    videos = sorted(f for d in folders if d.exists() for f in d.iterdir()
                    if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"))
    if videos:
        from .vision import analyze_video, video_events
        summary["videos"] = []
        for v in videos:
            print(f"[video] Analisando {v.name} (pose + objetos)...")
            res = analyze_video(v)
            events += video_events(res, config.RELATIVE_EPOCH)
            ang = res["angles"].set_index("time_s")
            fig, ax = plt.subplots(figsize=(8, 2.8))
            for c in [c for c in ang.columns if c.endswith(("knee", "hip", "shoulder", "elbow"))]:
                ax.plot(ang.index, ang[c], lw=1, label=c)
            ax.set_xlabel("seconds"); ax.set_ylabel("angle (°)"); ax.legend(fontsize=7, ncol=4)
            imgs = [fig_to_b64(fig)] + [file_to_b64(p) for p in res["snapshots"].values()]
            rp = write_report(f"video_{v.stem}.html", f"Physiotherapy video report — {v.name}", [
                {"heading": "Summary", "table": pd.DataFrame([res["summary"]])},
                {"heading": "Detected deviations / critical-area events",
                 "table": pd.DataFrame(res["findings"]) if res["findings"] else None,
                 "text": "" if res["findings"] else "No deviations detected."},
                {"heading": "Joint angles and annotated frames", "images": imgs},
            ])
            summary["videos"].append({**res["summary"], "findings": len(res["findings"]), "report": str(rp)})
    else:
        print(f"[video] Nenhum vídeo (.mp4/.avi/.mov/.mkv) em {config.PHYSIO_VIDEO_DIR} ou {config.DATASETS / 'physio_videos'} (OpenPose/YOLOv8 não executados)")

    clear_events("video")
    save_events(events)
    summary["events"] = len(events)
    (config.RESULTS_DIR / "video_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
