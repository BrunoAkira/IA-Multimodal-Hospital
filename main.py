"""
Hospital AI — monitoramento multimodal de pacientes.

Comandos:
    python main.py inspect     # confere datasets, vídeos, consultas e chaves do Azure
    python main.py clinical    # base clínica: sinais vitais, prescrições, transferências
    python main.py audio       # base de fala + consultas: voz, Speech to Text, Text Analytics
    python main.py video       # captura de movimento (.amc) + vídeos .mp4 (OpenPose / YOLOv8)
    python main.py video-validation  # valida a análise de vídeo contra um gabarito correto/incorreto
    python main.py fusion      # pacientes virtuais + fusão + alertas
    python main.py all         # tudo acima, em ordem
    python main.py monitor     # simulação de tempo real dos sinais vitais no terminal
    python main.py figures     # gráficos dos resultados para o relatório (results/figures)
    python main.py summary     # métricas principais de cada módulo no terminal
    python main.py alerts      # fila de alertas (--level critical --limit 10)
    python main.py consultations  # resultado da análise de cada consulta
    streamlit run dashboard/app.py
"""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Hospital AI — multimodal patient monitoring")
    parser.add_argument("command", choices=["inspect", "clinical", "audio", "video", "video-validation", "fusion", "all", "monitor", "figures",
                                            "summary", "alerts", "consultations"])
    parser.add_argument("--exercises", default="1,6", help="video-validation: ids dos exercícios (1=arm abduction ... 6=squats)")
    parser.add_argument("--max-videos", type=int, default=4, help="video-validation: máximo de vídeos por exercício")
    parser.add_argument("--rows", type=int, default=None, help="máximo de falas da base de fala")
    parser.add_argument("--speed", type=float, default=0.2, help="segundos reais por hora simulada (monitor)")
    parser.add_argument("--level", choices=["medium", "high", "critical"], default=None, help="alerts: mostra só um nível")
    parser.add_argument("--limit", type=int, default=10, help="alerts: quantos alertas mostrar")
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
        print(json.dumps(run_audio(args.rows or config.MAX_SPEECH_ROWS), indent=2, default=str))

    if args.command in ("video", "all"):
        from src.video.pipeline import run_video
        print(json.dumps(run_video(), indent=2, default=str))

    if args.command == "video-validation":
        from src.video.video_validation import run_video_validation
        ex = [int(x) for x in args.exercises.split(",") if x.strip()]
        print(json.dumps(run_video_validation(ex, args.max_videos), indent=2, default=str))

    if args.command in ("fusion", "all"):
        from src.fusion import run_fusion
        run_fusion()

    if args.command == "summary":
        from src.results_view import print_summary
        print_summary()

    if args.command == "alerts":
        from src.results_view import print_alerts
        print_alerts(args.level, args.limit)

    if args.command == "consultations":
        from src.results_view import print_consultations
        print_consultations()

    if args.command == "figures":
        from src.report_figures import run_figures
        print(json.dumps(run_figures(), indent=2))

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
