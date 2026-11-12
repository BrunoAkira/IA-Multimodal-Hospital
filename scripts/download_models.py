"""
Baixa os pesos dos modelos de visão para a pasta models/:

  * YOLOv8n (detecção de objetos, ~6 MB) e YOLOv8n-pose (~7 MB) — Ultralytics
  * OpenPose COCO (18 pontos, ~200 MB) — CMU Perceptual Computing Lab

Se o download do OpenPose falhar, a estimativa de pose usa o YOLOv8-pose.
Licenças: OpenPose = uso acadêmico/não comercial; YOLOv8 = AGPL-3.0.
"""
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

OPENPOSE_FILES = {
    "pose_deploy_linevec.prototxt":
        "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt",
    "pose_iter_440000.caffemodel":
        "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel",
}


def download(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print("já existe:", dest.name)
        return True
    print("baixando", dest.name, "...")
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as exc:
        print(f"  falhou ({exc}).")
        if dest.exists():
            dest.unlink()
        return False


if __name__ == "__main__":
    # YOLOv8: a própria biblioteca baixa os pesos
    from ultralytics import YOLO
    for name in ("yolov8n.pt", "yolov8n-pose.pt"):
        target = config.MODELS_DIR / name
        if not target.exists():
            YOLO(name)                      # baixa para a pasta atual
            Path(name).replace(target)      # move para models/
        print("ok:", target)

    op = config.MODELS_DIR / "openpose"
    op.mkdir(parents=True, exist_ok=True)
    ok = all(download(url, op / fname) for fname, url in OPENPOSE_FILES.items())
    print("OpenPose pronto." if ok else "OpenPose indisponível: será usado YOLOv8-pose.")
