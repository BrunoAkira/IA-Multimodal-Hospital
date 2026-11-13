"""
Exemplo: termos críticos e negação num texto de consulta.

    python examples/text_terms_example.py "No fever. I have chest pain and I feel short of breath."
    python examples/text_terms_example.py "..." --azure     # usa o Azure Text Analytics for health

Sem --azure, a análise é local (dicionário de termos), sem gastar cota.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.audio.text_analysis import local_critical_terms
from src.core import azure_client

parser = argparse.ArgumentParser()
parser.add_argument("text", nargs="?", default="No fever. I have chest pain and I feel short of breath.")
parser.add_argument("--azure", action="store_true", help="usa o Azure (1 registro da cota na primeira vez)")
args = parser.parse_args()

print("Texto:", args.text, "\n")
if args.azure:
    result = azure_client.analyze_health(args.text)
    if result is None:
        raise SystemExit("Azure Language indisponível: confira as chaves no .env")
    for e in result["entities"]:
        print(f"{e['text']:25} {e['category']:20} {e.get('assertion') or 'relatado'}")
else:
    for t in local_critical_terms(args.text):
        print(f"{t['text']:25} {'negado' if t['negated'] else 'relatado':10} peso {t['weight']}")
