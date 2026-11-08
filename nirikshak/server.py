"""Operator dashboard: stdlib HTTP server, no extra dependencies.

GET /              page (polls the API every second)
GET /api/recent    last inspections with advice
GET /api/stats     counts and latency per decision, line trend
GET /frame.jpg     latest annotated frame
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .store import Store

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nirikshak QC</title><style>
body{font:14px system-ui,sans-serif;margin:0;background:#111;color:#eee}header{padding:10px 16px;background:#1b1b1b}
main{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:12px;padding:12px}
img{width:100%;border-radius:6px;background:#000}.k{display:flex;gap:10px;margin:8px 0}
.k div{flex:1;background:#1e1e1e;border-radius:6px;padding:8px}.k b{display:block;font-size:20px}
.ev{background:#1e1e1e;border-radius:6px;padding:8px;margin-bottom:8px;border-left:4px solid #888}
.REJECT{border-color:#e33}.ESCALATE{border-color:#fa0}.PASS{border-color:#3b3}small{color:#aaa}
@media(max-width:800px){main{grid-template-columns:1fr}}</style></head><body>
<header><b>Nirikshak</b> on-device quality control</header><main><section><img id="f" src="/frame.jpg">
<div class="k" id="k"></div></section><section id="ev"></section></main><script>
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function tick(){try{
 document.getElementById('f').src='/frame.jpg?'+Date.now();
 const s=await (await fetch('/api/stats')).json(),r=await (await fetch('/api/recent')).json();
 const c=d=>(s.decisions[d]||{}).count||0;
 document.getElementById('k').innerHTML=`<div>PASS<b>${c('PASS')}</b></div><div>REJECT<b>${c('REJECT')}</b></div>
 <div>ESCALATE<b>${c('ESCALATE')}</b></div><div>defect rate<b>${(100*s.trend.defect_rate).toFixed(1)}%</b></div>`;
 document.getElementById('ev').innerHTML=r.filter(e=>e.decision!=='PASS').slice(0,12).map(e=>{const a=e.advice;
  return `<div class="ev ${e.decision}"><b>${e.decision}</b> ${esc(e.top_label)} · ${esc(e.top_severity)}
  <small>p=${e.p_defect} · ${e.latency_ms} ms · ${esc(e.frame)}</small><br>${a?`<b>${esc(a.action)}</b> (${esc(a.urgency)}) ${esc(a.root_cause)}<br>
  <small>Check: ${esc(a.operator_check)} · ${esc(a.source)}${a.stats&&a.stats.decode_tok_s?' · '+a.stats.decode_tok_s.toFixed(1)+' tok/s':''}</small>`:'<small>advisor running…</small>'}</div>`}).join('');
}catch(e){}}setInterval(tick,1000);tick();</script></body></html>"""


class LiveState:
    def __init__(self) -> None:
        self.jpeg: bytes = b""
        self.lock = threading.Lock()

    def set(self, jpeg: bytes) -> None:
        with self.lock:
            self.jpeg = jpeg

    def get(self) -> bytes:
        with self.lock:
            return self.jpeg


def serve(store: Store, live: LiveState, host: str = "0.0.0.0", port: int = 8000,
          trend_window: int = 50) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a) -> None:  # quiet
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/recent":
                self._send(json.dumps(store.recent(50)).encode(), "application/json")
            elif path == "/api/stats":
                body = {"decisions": store.stats(), "trend": store.trend(trend_window)}
                self._send(json.dumps(body).encode(), "application/json")
            elif path == "/frame.jpg" and live.get():
                self._send(live.get(), "image/jpeg")
            else:
                self._send(b"not found", "text/plain", 404)

    httpd = ThreadingHTTPServer((host, port), H)
    threading.Thread(target=httpd.serve_forever, daemon=True, name="dashboard").start()
    return httpd
