"""
(Opcional) Gera os .wav das consultas a partir dos roteiros, com voz sintética OFFLINE.

Usa pyttsx3 (gratuito, usa as vozes do próprio sistema operacional: SAPI no Windows,
NSSpeech no macOS, espeak no Linux). Nada vai para a internet.

    pip install pyttsx3
    python scripts/synthesize_consultations.py

Dica: gravar com a voz do grupo (celular -> .wav) deixa a demonstração mais realista.
Use falas mais lentas/cansadas nos roteiros 04 e 11 para testar os indicadores vocais.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

try:
    import pyttsx3
except ImportError:
    sys.exit("Instale antes: pip install pyttsx3")

engine = pyttsx3.init()
# escolhe uma voz em inglês, se existir
for v in engine.getProperty("voices"):
    if "en" in str(v.languages).lower() or "english" in v.name.lower():
        engine.setProperty("voice", v.id)
        break

for txt in sorted(config.CONSULT_DIR.glob("consult_*.txt")):
    wav = txt.with_suffix(".wav")
    if wav.exists():
        continue
    # só as falas, sem "Doctor:" / "Patient:"
    text = " ".join(l.split(":", 1)[-1].strip() for l in txt.read_text(encoding="utf-8").splitlines())
    # consultas com cansaço/falta de ar ficam mais lentas
    engine.setProperty("rate", 120 if txt.stem.endswith(("04", "11")) else 170)
    engine.save_to_file(text, str(wav))
    print("gerado:", wav.name)
engine.runAndWait()
