"""Render reports/PROJECT_REPORT.md to a self-contained HTML file (images embedded)."""
import base64, re
from pathlib import Path
import markdown

root = Path(__file__).resolve().parents[1]
md = (root / "reports" / "PROJECT_REPORT.md").read_text()
html = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])

def embed(m):
    p = (root / "reports" / m.group(1)).resolve()
    return 'src="data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode() + '"'

html = re.sub(r'src="(\.\./[^"]+\.png)"', embed, html)
css = """body{font-family:Calibri,Arial,sans-serif;max-width:980px;margin:30px auto;padding:0 18px;color:#1b2a41;line-height:1.5}
h1{color:#1b2a41;border-bottom:3px solid #e07a1f}h2{color:#1f5f8b;margin-top:34px;border-bottom:1px solid #d5dde5}
table{border-collapse:collapse;margin:12px 0;font-size:14px}th,td{border:1px solid #d5dde5;padding:5px 9px;text-align:left}
th{background:#f3f6f9}img{max-width:100%}code{background:#f3f6f9;padding:1px 4px}blockquote{border-left:4px solid #e07a1f;margin-left:0;padding-left:12px;color:#445}"""
(root / "reports" / "PROJECT_REPORT.html").write_text(f"<html><head><meta charset='utf-8'><title>FreshBasket forecasting report</title><style>{css}</style></head><body>{html}</body></html>")
print("html written")
