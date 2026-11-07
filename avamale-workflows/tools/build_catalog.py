"""Create an offline static catalog linking frozen synthetic review reports."""
from pathlib import Path
import html
from avamale_suite.workflows import SPEC, SENSITIVE_SYNTHETIC_ONLY

ROOT=Path(__file__).resolve().parents[1]
CARDS=[]
for code,(title,_) in SPEC.items():
    caution='Sensitive / synthetic-only' if code in SENSITIVE_SYNTHETIC_ONLY else ('Priority pilot candidate' if code in {'AA02','AA03'} else 'Structured-review foundation')
    CARDS.append(f'''<a class="card" href="demo_outputs/{code}/review.html"><span class="code">{code}</span><h2>{html.escape(title)}</h2><small>{html.escape(caution)} · Open synthetic report</small></a>''')
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>AVAMALE — 18 synthetic workflow demos</title>
<style>body{font:16px/1.5 system-ui,sans-serif;color:#18353e;background:#f3f7f6;margin:0}main{max-width:1160px;margin:auto;padding:42px 22px}header{font-size:13px;font-weight:800;letter-spacing:.15em}h1{font-size:clamp(28px,5vw,48px);line-height:1.1;margin:20px 0 10px}p{max-width:800px;color:#4d6870}.warning{margin:22px 0;background:#fff3de;padding:16px 20px;border-left:4px solid #c38622}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:14px}.card{display:block;text-decoration:none;color:inherit;background:white;border:1px solid #d8e5e2;border-radius:12px;padding:22px;transition:transform .2s}.card:hover,.card:focus-visible{transform:translateY(-2px);outline:2px solid #316a73}.code{display:inline-block;background:#d9ece8;padding:4px 8px;border-radius:5px;font-weight:bold;font-size:13px}.card h2{font-size:17px;line-height:1.3;margin:14px 0}.card small{color:#58717b}footer{margin:30px 0;font-size:12px;color:#64747c}a:focus-visible{outline:3px solid #316a73}</style>
<main><header>AVAMALE / REVIEW LAB</header><h1>18 workflows. Explicit rules.<br>Human decisions.</h1><p>Review our synthetic demonstrations and assess where your team still spends time checking records, exceptions and supporting evidence.</p><div class="warning"><strong>Engineering prototypes only.</strong> These demonstrations use synthetic data and are not evidence of production readiness, real customer savings, security certification or professional approval. Do not upload customer data.</div><p><a href="Review_Lab_AA02_AA03.html">Open the interactive AA02 / AA03 browser demo</a></p><section class="grid">''' + ''.join(CARDS) + '''</section><footer>Offline static catalog: no data collection, analytics, model calls or external assets. For private, approved feedback use the CLI and consent process in the repository documentation.</footer></main></html>'''
(ROOT/'catalog.html').write_text(page,encoding='utf-8')
print('Wrote static catalog with',len(CARDS),'workflows')
