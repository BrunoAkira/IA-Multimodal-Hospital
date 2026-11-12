"""
Visão computacional em vídeos de fisioterapia (.mp4, .avi, .mov, .mkv).

  * OpenPose (via OpenCV DNN, roda em CPU) -> 18 pontos do corpo por quadro
      - se os pesos do OpenPose não estiverem em models/openpose/, usamos
        YOLOv8-pose como alternativa (mesma saída: pontos do corpo)
  * YOLOv8 (ultralytics) -> detecção de pessoas/objetos e ÁREAS CRÍTICAS
      - patient_left_exercise_area: paciente fora da área do exercício
      - obstacle_in_critical_area: objeto no chão na área de risco de queda
      - patient_out_of_view: paciente sumiu da câmera

A partir dos pontos calculamos ângulos (joelho, quadril, ombro, cotovelo) e
procuramos: amplitude reduzida, assimetria esquerda/direita, repetições
irregulares e possível queda.

Processamento: ~5 quadros por segundo, redimensionados para 640 px de largura,
um quadro por vez (o vídeo não é carregado inteiro na memória).
O prefixo do nome do arquivo define o exercício (ex.: squat_patient01.mp4).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .. import config
from ..core.events import Event

OPENPOSE_DIR = config.MODELS_DIR / "openpose"
OPENPOSE_PROTO = OPENPOSE_DIR / "pose_deploy_linevec.prototxt"
OPENPOSE_WEIGHTS = OPENPOSE_DIR / "pose_iter_440000.caffemodel"

# Índices dos pontos em cada modelo -> nomes padronizados (inglês)
OPENPOSE_COCO = {0: "nose", 1: "neck", 2: "r_shoulder", 3: "r_elbow", 4: "r_wrist", 5: "l_shoulder",
                 6: "l_elbow", 7: "l_wrist", 8: "r_hip", 9: "r_knee", 10: "r_ankle", 11: "l_hip",
                 12: "l_knee", 13: "l_ankle"}
YOLO_POSE = {0: "nose", 5: "l_shoulder", 6: "r_shoulder", 7: "l_elbow", 8: "r_elbow", 9: "l_wrist",
             10: "r_wrist", 11: "l_hip", 12: "r_hip", 13: "l_knee", 14: "r_knee", 15: "l_ankle", 16: "r_ankle"}
SKELETON = [("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
            ("l_shoulder", "r_shoulder"), ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"), ("l_hip", "r_hip"),
            ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("r_hip", "r_knee"), ("r_knee", "r_ankle")]

# ângulo da articulação = ângulo no ponto do meio (ex.: joelho entre quadril e tornozelo)
ANGLES = {"knee": ("hip", "knee", "ankle"), "hip": ("shoulder", "hip", "knee"),
          "shoulder": ("hip", "shoulder", "elbow"), "elbow": ("shoulder", "elbow", "wrist")}

# Exercícios conhecidos: articulação principal e amplitude mínima esperada (graus)
PHYSIO_EXERCISES = {
    "squat": {"joint": "knee", "min_rom": 50},
    "sit_to_stand": {"joint": "knee", "min_rom": 60},
    "knee_extension": {"joint": "knee", "min_rom": 50},
    "arm_raise": {"joint": "shoulder", "min_rom": 100},
    "elbow_flexion": {"joint": "elbow", "min_rom": 90},
}

# Zonas padrão em coordenadas normalizadas (0..1) da imagem; um <video>.zones.json substitui por vídeo.
ZONES = {
    "exercise_area": np.array([[0.15, 0.0], [0.85, 0.0], [0.85, 1.0], [0.15, 1.0]]),
    "fall_risk_floor": np.array([[0.0, 0.85], [1.0, 0.85], [1.0, 1.0], [0.0, 1.0]]),
}
FLOOR_OBSTACLES = {"chair", "bottle", "backpack", "handbag", "suitcase", "sports ball", "cup", "book", "bench"}


# ---------------------------------------------------------------------------
# Estimadores de pose
# ---------------------------------------------------------------------------
class PoseEstimator:
    def __init__(self):
        if OPENPOSE_PROTO.exists() and OPENPOSE_WEIGHTS.exists():
            self.backend = "openpose"
            self.net = cv2.dnn.readNetFromCaffe(str(OPENPOSE_PROTO), str(OPENPOSE_WEIGHTS))
        else:
            from ultralytics import YOLO
            self.backend = "yolov8-pose"
            self.net = YOLO(str(config.MODELS_DIR / "yolov8n-pose.pt"))
        print(f"[video] Estimador de pose: {self.backend}")

    def __call__(self, frame: np.ndarray) -> dict:
        """Retorna {nome_do_ponto: (x, y, confiança)} com x,y normalizados 0..1."""
        if self.backend == "openpose":
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
            self.net.setInput(blob)
            out = self.net.forward()                 # (1, 57, 46, 46): mapas de calor + PAFs
            h, w = out.shape[2], out.shape[3]
            pts = {}
            for i, name in OPENPOSE_COCO.items():
                _, conf, _, (x, y) = cv2.minMaxLoc(out[0, i])   # ponto mais "quente" = articulação
                if conf > 0.1:
                    pts[name] = (x / w, y / h, float(conf))
            return pts
        res = self.net.predict(frame, imgsz=640, conf=0.3, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            return {}
        best = int(res.boxes.conf.argmax())          # pessoa mais confiável (o paciente)
        xy = res.keypoints.xyn[best].cpu().numpy()
        cf = res.keypoints.conf[best].cpu().numpy() if res.keypoints.conf is not None else np.ones(len(xy))
        return {name: (float(xy[i, 0]), float(xy[i, 1]), float(cf[i]))
                for i, name in YOLO_POSE.items() if cf[i] > 0.3}


def joint_angle(a, b, c) -> float:
    """Ângulo ABC em graus (b é o vértice)."""
    ba, bc = np.array(a[:2]) - np.array(b[:2]), np.array(c[:2]) - np.array(b[:2])
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def frame_angles(pts: dict) -> dict:
    out = {}
    for joint, (p1, p2, p3) in ANGLES.items():
        for side in ("l", "r"):
            names = [f"{side}_{p}" for p in (p1, p2, p3)]
            if all(n in pts for n in names):
                out[f"{side}_{joint}"] = joint_angle(*(pts[n] for n in names))
    # inclinação do tronco: vetor meio-dos-ombros -> meio-do-quadril vs vertical
    need = ("l_shoulder", "r_shoulder", "l_hip", "r_hip")
    if all(n in pts for n in need):
        sh = (np.array(pts["l_shoulder"][:2]) + pts["r_shoulder"][:2]) / 2
        hp = (np.array(pts["l_hip"][:2]) + pts["r_hip"][:2]) / 2
        v = sh - hp
        out["trunk_lean"] = float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]) + 1e-9)))
        out["hip_y"] = float(hp[1])
    return out


# ---------------------------------------------------------------------------
# Análise das séries de ângulos (independe do modelo usado)
# ---------------------------------------------------------------------------
def analyze_angles(angles: pd.DataFrame, exercise: str | None, fps: float) -> tuple[list[dict], dict]:
    findings, summary = [], {}
    spec = PHYSIO_EXERCISES.get(exercise or "", None)
    joints = [spec["joint"]] if spec else ["knee", "hip", "shoulder", "elbow"]

    for joint in joints:
        l, r = angles.get(f"l_{joint}"), angles.get(f"r_{joint}")
        if l is None or r is None or l.notna().sum() < 10:
            continue
        l, r = l.interpolate(limit=5), r.interpolate(limit=5)
        rom_l, rom_r = float(l.quantile(0.95) - l.quantile(0.05)), float(r.quantile(0.95) - r.quantile(0.05))
        rom = max(rom_l, rom_r)
        summary[f"{joint}_rom_left"], summary[f"{joint}_rom_right"] = round(rom_l, 1), round(rom_r, 1)
        if spec and rom < spec["min_rom"]:
            findings.append({"event_type": "reduced_range_of_motion", "score": min(1.0, 0.5 + (spec["min_rom"] - rom) / spec["min_rom"]),
                             "message": f"{joint} range of motion {rom:.0f}° (expected ≥ {spec['min_rom']}°)"})
        # exercício unilateral (ex.: abdução de UM braço): um lado quase parado.
        # Nesse caso não é assimetria, e a análise usa só o lado que se move.
        unilateral = rom > 20 and min(rom_l, rom_r) < 0.3 * rom
        summary[f"{joint}_unilateral"] = bool(unilateral)
        if rom > 20 and not unilateral and abs(rom_l - rom_r) / rom > 0.25:
            findings.append({"event_type": "left_right_asymmetry", "score": 0.55,
                             "message": f"{joint} ROM left {rom_l:.0f}° vs right {rom_r:.0f}°"})
        # série usada para contar: o lado que se move (unilateral) ou a média dos dois
        if unilateral:
            series = l if rom_l >= rom_r else r
        else:
            series = (l + r) / 2
        # joelho (agachamento/avanço): de frente, o ângulo 2D do joelho quase não muda;
        # a altura do quadril sobe e desce em qualquer ângulo de câmera -> conta por ela
        by_hip = joint == "knee" and "hip_y" in angles and angles["hip_y"].notna().sum() > 10
        if by_hip:
            series = -angles["hip_y"].interpolate(limit=5)   # quadril mais baixo = y maior
            summary[f"{joint}_count_signal"] = "hip_height"
        # suaviza (mediana móvel de ~0,6 s) para o ruído não virar repetição extra
        win = max(3, int(0.6 * fps) | 1)
        series = series.bfill().ffill().rolling(win, center=True, min_periods=1).median().to_numpy()
        if rom > 20 or by_hip:
            # ombro: a repetição é o ângulo SUBINDO (pico); joelho/quadril/cotovelo: flexão (vale)
            signal = series if joint == "shoulder" else -series
            amp = float(np.nanpercentile(series, 95) - np.nanpercentile(series, 5))
            peaks, _ = find_peaks(signal, prominence=0.5 * amp, distance=max(1, int(1.0 * fps)))
            summary[f"{joint}_repetitions"] = int(len(peaks))
            if len(peaks) >= 3:
                durations = np.diff(peaks) / fps
                # intervalos > 2x a mediana são pausas entre blocos de exercício, não repetições lentas
                durations = durations[durations <= 2 * np.median(durations)]
                cv = float(durations.std() / durations.mean()) if len(durations) >= 2 else 0.0
                summary[f"{joint}_rep_duration_cv"] = round(cv, 2)
                if cv > 0.35:
                    findings.append({"event_type": "irregular_repetitions", "score": min(1.0, 0.4 + cv / 2),
                                     "message": f"Irregular repetition timing (CV {cv:.2f})"})

    # possível queda: quadril desce muito rápido e o tronco fica quase horizontal
    if "hip_y" in angles and angles["hip_y"].notna().sum() > fps:
        drop = angles["hip_y"].diff(int(max(1, fps))).max()     # descida em ~1 s
        lean = angles["trunk_lean"].max()
        summary["max_trunk_lean"] = round(float(lean), 1)
        if drop > 0.25 and lean > 60:
            findings.append({"event_type": "possible_fall", "score": 0.95,
                             "message": "Rapid hip drop with horizontal trunk (possible fall)"})
    return findings, summary


# ---------------------------------------------------------------------------
# Vídeo completo: pose + objetos + zonas
# ---------------------------------------------------------------------------
def load_zones(video: Path) -> dict:
    """
    Zonas do vídeo: usa '<nome_do_video>.zones.json' (mesma pasta) se existir, senão ZONES.
    Exemplo de arquivo:  {"exercise_area": [[0.3,0],[0.9,0],[0.9,1],[0.3,1]], "fall_risk_floor": null}
    (null desliga aquela zona)
    """
    import json
    zones = dict(ZONES)
    f = video.with_suffix(".zones.json")
    if f.exists():
        for k, v in json.loads(f.read_text(encoding="utf-8")).items():
            if v is None:
                zones.pop(k, None)
            else:
                zones[k] = np.array(v, dtype=float)
        print(f"[video] Zonas personalizadas: {f.name}")
    return zones


def _in_zone(zones: dict, zone: str, x: float, y: float) -> bool:
    if zone not in zones:      # zona desligada
        return zone == "exercise_area"
    return cv2.pointPolygonTest(zones[zone].astype(np.float32), (x, y), False) >= 0


def _draw(frame, pts, boxes, zones=ZONES):
    h, w = frame.shape[:2]
    for name, poly in zones.items():
        cv2.polylines(frame, [(poly * [w, h]).astype(np.int32)], True, (0, 200, 255), 1)
    for a, b in SKELETON:
        if a in pts and b in pts:
            cv2.line(frame, (int(pts[a][0] * w), int(pts[a][1] * h)), (int(pts[b][0] * w), int(pts[b][1] * h)),
                     (0, 255, 0), 2)
    for (x1, y1, x2, y2, label) in boxes:
        cv2.rectangle(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (255, 120, 0), 1)
        cv2.putText(frame, label, (int(x1 * w), int(y1 * h) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1)
    return frame


def analyze_video(path: Path, target_fps: float = 5.0, max_seconds: float = 300,
                  detect_objects: bool = True) -> dict:
    """
    detect_objects=False pula o YOLOv8 de objetos/zonas (só pose): ~2x mais rápido,
    usado na avaliação em lote do REHAB24-6.
    """
    from ultralytics import YOLO

    # aceita "arm_raise_x.mp4", "ArmRaiseR3S01.avi", "MiniSquat...", "SitToStand..."
    name = path.stem.lower().replace("_", "").replace("-", "")
    aliases = {"minisquat": "squat", "squat": "squat", "sittostand": "sit_to_stand",
               "kneeextension": "knee_extension", "armraise": "arm_raise", "armabduction": "arm_raise",
               "elbowflexion": "elbow_flexion"}
    exercise = next((ex for key, ex in aliases.items() if name.startswith(key)), None)
    pose = PoseEstimator()
    detector = YOLO(str(config.MODELS_DIR / "yolov8n.pt")) if detect_objects else None
    zones = load_zones(path)
    snap_dir = config.REPORTS_DIR / "video_frames"
    snap_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(src_fps / target_fps))
    fps = src_fps / step
    rows, zone_counts, findings, snapshots = [], {"outside": 0, "missing": 0}, [], {}
    zone_flags = set()
    i = 0
    while True:
        if i % step:                       # quadro pulado: só avança (bem mais rápido que ler)
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok or i / src_fps > max_seconds:
            break
        frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
        t = i / src_fps
        pts = pose(frame)
        rows.append({"time_s": t, "frame": i, **frame_angles(pts)})   # frame = índice no vídeo original

        if detector is None:               # modo só pose
            if len(rows) == 1:
                snapshots["pose"] = _draw(frame.copy(), pts, [], zones)
            i += 1
            continue

        # YOLOv8: pessoas e objetos
        det = detector.predict(frame, imgsz=640, conf=0.4, verbose=False)[0]
        boxes, person_feet = [], None
        for (x1, y1, x2, y2), c in zip(det.boxes.xyxyn.cpu().numpy(), det.boxes.cls.cpu().numpy()):
            label = det.names[int(c)]
            boxes.append((x1, y1, x2, y2, label))
            if label == "person" and person_feet is None:
                person_feet = ((x1 + x2) / 2, y2)          # ponto dos pés = base da caixa
            elif label in FLOOR_OBSTACLES and _in_zone(zones, "fall_risk_floor", (x1 + x2) / 2, y2):
                if "obstacle" not in zone_flags:
                    zone_flags.add("obstacle")
                    findings.append({"event_type": "obstacle_in_critical_area", "score": 0.6, "time_s": t,
                                     "message": f"'{label}' on the floor in the fall-risk area"})
                    snapshots["obstacle"] = _draw(frame.copy(), pts, boxes, zones)

        # paciente fora da área / fora da câmera (precisa durar > 2-3 s)
        zone_counts["missing"] = zone_counts["missing"] + 1 if person_feet is None else 0
        outside = person_feet is not None and not _in_zone(zones, "exercise_area", *person_feet)
        zone_counts["outside"] = zone_counts["outside"] + 1 if outside else 0
        if zone_counts["missing"] > 3 * fps and "missing" not in zone_flags:
            zone_flags.add("missing")
            findings.append({"event_type": "patient_out_of_view", "score": 0.5, "time_s": t,
                             "message": "Patient not visible for more than 3 s"})
        if zone_counts["outside"] > 2 * fps and "outside" not in zone_flags:
            zone_flags.add("outside")
            findings.append({"event_type": "patient_left_exercise_area", "score": 0.55, "time_s": t,
                             "message": "Patient outside the exercise area for more than 2 s"})
            snapshots["outside"] = _draw(frame.copy(), pts, boxes, zones)
        if len(rows) == 1 or (len(rows) % int(10 * fps) == 0 and "pose" not in snapshots):
            snapshots["pose"] = _draw(frame.copy(), pts, boxes, zones)
        i += 1
    cap.release()

    angles = pd.DataFrame(rows)
    angle_findings, summary = analyze_angles(angles, exercise, fps)
    findings += angle_findings
    summary.update({"video": path.name, "exercise": exercise or "unknown", "pose_backend": pose.backend,
                    "frames_analyzed": len(angles),
                    "pose_visible_fraction": round(float(angles.drop(columns=["time_s", "frame"]).notna().any(axis=1).mean()), 2)
                    if len(angles) else 0.0})
    snap_files = {}
    for k, img in snapshots.items():
        f = snap_dir / f"{path.stem}_{k}.jpg"
        cv2.imwrite(str(f), img)
        snap_files[k] = str(f)
    angles.to_csv(config.RESULTS_DIR / f"video_angles_{path.stem}.csv", index=False)
    return {"summary": summary, "findings": findings, "angles": angles, "snapshots": snap_files}


def video_events(result: dict, base_ts: str) -> list[Event]:
    stem = Path(result["summary"]["video"]).stem
    t0 = pd.Timestamp(base_ts) + pd.Timedelta(hours=15)
    return [Event(f"physio:{stem}", (t0 + pd.Timedelta(seconds=f.get("time_s", 0))).isoformat(), "video",
                  "physio_vision", f["event_type"], f["score"], message=f["message"],
                  evidence={"video": result["summary"]["video"], "exercise": result["summary"]["exercise"]})
            for f in result["findings"]]
