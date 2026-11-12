"""
Hospital AI — monitoramento multimodal (protótipo acadêmico).

Comandos:
    python main.py inspect     # confere datasets, vídeos, consultas e chaves do Azure
    python main.py clinical    # MIMIC: sinais vitais, prescrições, transferências
    python main.py audio       # TORGO + consultas: voz, Speech to Text, Text Analytics
    python main.py video       # UL-RED (.amc) + vídeos .mp4 (OpenPose / YOLOv8)
    python main.py rehab24     # avaliação em vídeo real com o gabarito do REHAB24-6
    python main.py fusion      # pacientes virtuais + fusão + alertas
    python main.py all         # tudo acima, em ordem
    python main.py monitor     # simulação de tempo real dos sinais vitais no terminal
    streamlit run dashboard/app.py
"""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Hospital AI — academic multimodal monitoring prototype")
    parser.add_argument("command", choices=["inspect", "clinical", "audio", "video", "rehab24", "fusion", "all", "monitor"])
    parser.add_argument("--exercises", default="1,6", help="rehab24: ids dos exercícios (1=arm abduction ... 6=squats)")
    parser.add_argument("--max-videos", type=int, default=4, help="rehab24: máximo de vídeos por exercício")
    parser.add_argument("--rows", type=int, default=None, help="máximo de falas do TORGO")
    parser.add_argument("--speed", type=float, default=0.2, help="segundos reais por hora simulada (monitor)")
    parser.add_argument("--hours", type=int, default=None, help="limite de horas simuladas (monitor; padrão = todas)")
    args = parser.parse_args()

    # imports dentro de cada comando: só carrega o que vai usar (mais rápido e leve)
    if args.command == "inspect":
        from src.inspect_data import inspect_all
        inspect_all()

    if args.command in ("clinical", "all"):
        from src.clinical.pipeline import run_clinical
        print(json.dumps(run_clinical(), indent=2, default=str))

    if args.command in ("audio", "all"):
        from src import config
        from src.audio.pipeline import run_audio
        print(json.dumps(run_audio(args.rows or config.MAX_TORGO_ROWS), indent=2, default=str))

    if args.command in ("video", "all"):
        from src.video.pipeline import run_video
        print(json.dumps(run_video(), indent=2, default=str))

    if args.command == "rehab24":
        from src.video.rehab24 import run_rehab24
        ex = [int(x) for x in args.exercises.split(",") if x.strip()]
        print(json.dumps(run_rehab24(ex, args.max_videos), indent=2, default=str))

    if args.command in ("fusion", "all"):
        from src.fusion import run_fusion
        run_fusion()

    if args.command == "monitor":
        import pandas as pd
        from src import config
        from src.clinical.vitals import replay
        path = config.RESULTS_DIR / "vitals_hourly.csv"
        if not path.exists():
            raise SystemExit("Rode antes: python main.py clinical")
        hourly = pd.read_csv(path, parse_dates=["charttime"])
        colors = {"high": "\033[91m", "medium": "\033[93m", "low": "\033[0m"}
        print("Monitorando em tempo real (Ctrl+C para sair)...\n")

        def show(ev):
            c = colors.get(ev.severity, "")
            print(f"{c}[{ev.timestamp}] {ev.severity.upper():6} {ev.patient_id:16} {ev.event_type:28} {ev.message}\033[0m")

        try:
            replay(hourly, speed_seconds=args.speed, on_event=show, limit_hours=args.hours)
        except KeyboardInterrupt:
            print("\nencerrado.")


if __name__ == "__main__":
    main()
