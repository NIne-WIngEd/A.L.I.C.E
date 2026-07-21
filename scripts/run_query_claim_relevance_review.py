from __future__ import annotations
import argparse, html, sys, urllib.parse, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from alice_vault.query_claim_relevance_calibration import load_query_claim_relevance_calibration_policy, load_relevance_calibration_bundle, save_relevance_human_label

def esc(v): return html.escape(str(v or ""))
def render(path,bundle):
    items=list(bundle.get("items",[])); labeled=sum(bool(str(x.get("relevance_human_label", ""))) for x in items); idx=next((i for i,x in enumerate(items) if not str(x.get("relevance_human_label", ""))),None)
    if idx is None:
        body=f"<h2>Complete</h2><p>You labeled {labeled} of {len(items)} items.</p><pre>{esc(path)}</pre>"
    else:
        item=items[idx]
        body=(f'<p>Item {idx+1} of {len(items)} - {labeled} labeled</p>'
              f'<h2>Question</h2><div class="box">{esc(item.get("question",""))}</div>'
              f'<h2>Candidate claim</h2><div class="box">{esc(item.get("claim_text",""))}</div>'
              '<h2>Review criteria</h2>'
              '<p><b>Relevant:</b> directly answers the question or provides a material fact needed to answer it.</p>'
              '<p><b>Partially relevant:</b> related, but only weakly, indirectly, or incompletely helps answer it.</p>'
              '<p><b>Irrelevant:</b> does not materially help answer the question, even if true.</p>'
              f'<form method="post" action="/label"><input type="hidden" name="item_id" value="{esc(item.get("item_id",""))}">'
              '<button name="label" value="relevant">Relevant</button><button name="label" value="partially_relevant">Partially relevant</button><button name="label" value="irrelevant">Irrelevant</button></form>')
    return '<!doctype html><meta charset="utf-8"><title>A.L.I.C.E. Relevance Calibration</title><style>body{font-family:sans-serif;max-width:900px;margin:40px auto;line-height:1.5}.box{white-space:pre-wrap;border:1px solid #ccc;padding:16px;border-radius:8px}button{margin:8px;padding:10px 16px}</style><h1>A.L.I.C.E. Query-Claim Relevance Calibration</h1>'+body
class Handler(BaseHTTPRequestHandler):
    bundle_path: Path
    def send_text(self,status,content,ctype="text/html; charset=utf-8"):
        data=content.encode("utf-8"); self.send_response(status); self.send_header("Content-Type",ctype); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path not in {"/","/index.html"}: return self.send_text(404,"Not found","text/plain; charset=utf-8")
        self.send_text(200,render(self.bundle_path,load_relevance_calibration_bundle(self.bundle_path)))
    def do_POST(self):
        if self.path!="/label": return self.send_text(404,"Not found","text/plain; charset=utf-8")
        raw=self.rfile.read(int(self.headers.get("Content-Length","0"))).decode("utf-8"); vals=urllib.parse.parse_qs(raw)
        try: save_relevance_human_label(bundle_path=self.bundle_path,item_id=vals.get("item_id",[""])[0],label=vals.get("label",[""])[0])
        except Exception as exc: return self.send_text(400,esc(f"{type(exc).__name__}: {exc}"),"text/plain; charset=utf-8")
        self.send_response(303); self.send_header("Location","/"); self.end_headers()
    def log_message(self,format,*args): return
p=argparse.ArgumentParser(); p.add_argument("--vault",required=True,type=Path); p.add_argument("--calibration",required=True,type=Path); p.add_argument("--host"); p.add_argument("--port",type=int); p.add_argument("--no-browser",action="store_true")
a=p.parse_args(); a.vault.expanduser().resolve(strict=True); path=a.calibration.expanduser().resolve(strict=True); policy=load_query_claim_relevance_calibration_policy(); host=a.host or policy.review_host; port=a.port or policy.review_port
if host not in {"127.0.0.1","localhost","::1"}: raise ValueError("Review server may only bind to loopback")
Handler.bundle_path=path; server=ThreadingHTTPServer((host,port),Handler); url=f"http://{host}:{port}/"; print("A.L.I.C.E. query-claim relevance review"); print(f"Bundle: {path}"); print(f"Open: {url}"); print("Press Ctrl+C when review is complete.")
if not a.no_browser: webbrowser.open(url)
try: server.serve_forever()
except KeyboardInterrupt: print("\nReview server stopped.")
finally: server.server_close()
