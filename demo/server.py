"""Server locale della demo LastCall: mostra il sito e genera offerte con l'AI su richiesta.

La chiave della Token Factory resta su questo PC (churn/.env) e non arriva mai al browser:
la pagina chiama POST /api/generate {"customer_id": "..."} e il server chiama l'LLM.
Uso (dalla cartella AMD):  python demo/server.py   poi apri http://localhost:8766
"""
import json
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "churn"))

import pipeline  # noqa: E402  (stessa logica e stesso controllo delle offerte già generate)

PORT = 8766
print("Carico i clienti…")
CUSTOMERS = pipeline.load_customers().set_index("customer_id")
KEY = pipeline.api_key()
print(f"{len(CUSTOMERS)} clienti pronti. Apri http://localhost:{PORT}")


SAVE_FILE = HERE.parent / "data" / "generated_offers.json"
SAVE_LOCK = threading.Lock()
SAVED = json.loads(SAVE_FILE.read_text(encoding="utf-8")) if SAVE_FILE.exists() else {}


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")  # in locale la pagina deve sempre usare l'ultima versione
        super().end_headers()

    def send_json(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/health":
            return self.send_json(200, {"ok": True, "model": pipeline.MODEL})
        if self.path.startswith("/api/generated"):
            return self.send_json(200, list(SAVED.values()))
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/generate":
            return self.send_json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            cid = str(json.loads(self.rfile.read(length) or b"{}").get("customer_id", ""))
        except (ValueError, json.JSONDecodeError):
            return self.send_json(400, {"error": "bad request"})
        if cid not in CUSTOMERS.index:
            return self.send_json(404, {"error": "unknown customer"})
        row = CUSTOMERS.loc[cid].copy()
        row["customer_id"] = cid
        try:
            answer, _, attempts = pipeline.ask_valid(pipeline.to_state(row), KEY, tries=2, timeout=240)
        except Exception as e:  # rete, limiti della chiave, risposta non valida
            return self.send_json(502, {"error": f"AI request failed: {e}"})
        valid = pipeline.valid(answer, pipeline.to_state(row)["most_bought_product"])
        result = {
            "id": cid,
            "reason": answer.get("reason"),
            "offer": answer.get("offer"),
            "discount_pct": int(answer.get("discount_pct") or 0),
            "explanation": answer.get("explanation"),
            "message": answer.get("message"),
            "checks_ok": bool(valid),
            "net_spend": round(float(row["net_spend"]), 2),
            "orders": int(row["n_orders"]),
            "attempts": attempts,
        }
        with SAVE_LOCK:  # le offerte restano sul PC: la pagina le ricarica anche se generate fuori dal browser
            SAVED[cid] = result
            SAVE_FILE.write_text(json.dumps(SAVED, ensure_ascii=False, indent=1), encoding="utf-8")
        return self.send_json(200, result)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), partial(Handler, directory=str(HERE))).serve_forever()
