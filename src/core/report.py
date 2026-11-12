"""
Relatórios automáticos em HTML (um arquivo único, imagens embutidas em base64).
"""
from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # gera imagens sem abrir janela
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Template

from .. import config

TEMPLATE = Template("""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>{{ title }}</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;max-width:1000px;margin:24px auto;color:#222;padding:0 16px}
 h1{font-size:22px} h2{font-size:17px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}
 table{border-collapse:collapse;width:100%;font-size:13px} th,td{border:1px solid #ddd;padding:5px 7px;text-align:left}
 th{background:#f4f4f4} .high{color:#b00020;font-weight:600} .medium{color:#b26a00;font-weight:600}
 .low{color:#555} img{max-width:100%;border:1px solid #eee;margin:6px 0}
 .note{font-size:12px;color:#666;background:#fafafa;padding:8px;border-left:3px solid #ccc}
</style></head><body>
<h1>{{ title }}</h1>
<p>Generated {{ now }} — research prototype, <b>not for clinical decisions</b>.</p>
{% for s in sections %}
 <h2>{{ s.heading }}</h2>
 {% if s.text %}<p>{{ s.text }}</p>{% endif %}
 {% if s.table is not none %}{{ s.table }}{% endif %}
 {% for img in s.images %}<img src="data:image/{{ img.fmt }};base64,{{ img.b64 }}">{% endfor %}
{% endfor %}
<p class="note">Deviations are computed against the patient's own reference execution and
must be reviewed by a physiotherapist.</p>
</body></html>""")


def fig_to_b64(fig) -> dict:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {"fmt": "png", "b64": base64.b64encode(buf.getvalue()).decode()}


def file_to_b64(path: str) -> dict:
    return {"fmt": "jpeg", "b64": base64.b64encode(Path(path).read_bytes()).decode()}


def df_to_html(df: pd.DataFrame) -> str:
    html = df.to_html(index=False, border=0, float_format=lambda x: f"{x:.2f}", escape=True)
    for sev in ("high", "medium", "low"):
        html = html.replace(f"<td>{sev}</td>", f'<td class="{sev}">{sev}</td>')
    return html


def write_report(filename: str, title: str, sections: list[dict]) -> Path:
    """sections: [{"heading": str, "text": str, "table": DataFrame|None, "images": [..]}]"""
    for s in sections:
        s.setdefault("text", "")
        s.setdefault("images", [])
        t = s.get("table")
        s["table"] = df_to_html(t) if isinstance(t, pd.DataFrame) and not t.empty else None
    path = config.REPORTS_DIR / filename
    path.write_text(TEMPLATE.render(title=title, sections=sections,
                                    now=datetime.now().strftime("%Y-%m-%d %H:%M")), encoding="utf-8")
    return path
