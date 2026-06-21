"""
Export a markdown paper to a self-contained HTML + PDF.

No pandoc / LaTeX on this machine, so we render markdown -> HTML with python-markdown
(tables, fenced code), inline-embed the figures as base64 data URIs (fully portable
single file), apply an academic print stylesheet, then print to PDF via a headless
Chromium browser (Edge/Chrome) — which renders the Unicode (≈ × → ² σ ‖ …) and the
figures faithfully without any extra Python dependency.

Usage:
  dppo/Scripts/python.exe -m scripts.export_paper \
      --md docs/paper_negative_result_draft.md \
      --out docs/paper_negative_result_draft
  # writes <out>.html and (if a browser is found) <out>.pdf
"""
import os
import re
import sys
import base64
import argparse
import subprocess

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page { size: A4; margin: 18mm 16mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Georgia","Cambria","Times New Roman",serif; font-size: 10.5pt;
       line-height: 1.5; color: #111; max-width: 820px; margin: 0 auto; padding: 24px; }
h1 { font-size: 19pt; line-height: 1.25; margin: 0 0 6px; }
h2 { font-size: 14pt; border-bottom: 1.5px solid #ccc; padding-bottom: 3px; margin: 22px 0 8px; }
h3 { font-size: 12pt; margin: 16px 0 6px; }
h1,h2,h3 { font-family: "Helvetica Neue","Arial",sans-serif; color: #1a1a1a; page-break-after: avoid; }
p { margin: 7px 0; text-align: justify; }
code { font-family: "Consolas","Menlo",monospace; font-size: 9pt;
       background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }
pre { background: #f4f4f4; padding: 10px 12px; border-radius: 5px; overflow-x: auto;
      font-size: 8.7pt; line-height: 1.35; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 4px 7px; text-align: left; vertical-align: top; }
th { background: #efefef; font-family: "Helvetica Neue","Arial",sans-serif; }
tr:nth-child(even) td { background: #fafafa; }
img { max-width: 100%; display: block; margin: 12px auto; page-break-inside: avoid; }
blockquote { border-left: 3px solid #ccc; margin: 10px 0; padding: 2px 14px; color: #444; }
hr { border: none; border-top: 1px solid #ddd; margin: 18px 0; }
a { color: #1a5276; text-decoration: none; }
strong { color: #000; }
"""

HTML_TMPL = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title><style>{css}</style></head>
<body>{body}</body></html>
"""


def embed_images(html, base_dir):
    """Replace <img src="relative.png"> with base64 data URIs (portable single file)."""
    def repl(m):
        src = m.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        path = os.path.normpath(os.path.join(base_dir, src))
        if not os.path.exists(path):
            print(f"  [warn] image not found, left as link: {src}")
            return m.group(0)
        ext = os.path.splitext(path)[1].lstrip('.').lower() or 'png'
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('ascii')
        return f'src="data:image/{ext};base64,{b64}"'
    return re.sub(r'src="([^"]+)"', repl, html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', default='docs/paper_negative_result_draft.md')
    ap.add_argument('--out', default='docs/paper_negative_result_draft',
                    help='output basename (writes .html and .pdf)')
    ap.add_argument('--browser', default=None, help='path to chrome/edge (auto-detect if omitted)')
    ap.add_argument('--no-pdf', action='store_true', help='only write the HTML')
    args = ap.parse_args()

    md_path = os.path.join(ROOT, args.md)
    base_dir = os.path.dirname(md_path)
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()
    title = next((l[2:].strip() for l in text.splitlines() if l.startswith('# ')), 'Paper')

    body = markdown.markdown(
        text, extensions=['tables', 'fenced_code', 'sane_lists', 'attr_list', 'md_in_html'])
    body = embed_images(body, base_dir)
    html = HTML_TMPL.format(title=title, css=CSS, body=body)

    html_path = os.path.join(ROOT, args.out + '.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"wrote {html_path}  ({os.path.getsize(html_path)/1e6:.2f} MB, images embedded)")

    if args.no_pdf:
        return
    browser = args.browser or next((b for b in BROWSERS if os.path.exists(b)), None)
    if not browser:
        print("  no Chrome/Edge found — open the HTML and Print → Save as PDF.")
        return
    pdf_path = os.path.join(ROOT, args.out + '.pdf')
    url = 'file:///' + html_path.replace('\\', '/')
    cmd = [browser, '--headless', '--disable-gpu', '--no-pdf-header-footer',
           f'--print-to-pdf={pdf_path}', url]
    print(f"  rendering PDF via {os.path.basename(browser)} ...")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
        print(f"wrote {pdf_path}  ({os.path.getsize(pdf_path)/1e6:.2f} MB)")
    else:
        print(f"  PDF render failed (rc={r.returncode}). stderr:\n{r.stderr[-500:]}")
        print("  Fallback: open the HTML and Print → Save as PDF.")


if __name__ == '__main__':
    main()
