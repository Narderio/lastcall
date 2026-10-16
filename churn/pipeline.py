"""LastCall: dal punteggio del Random Forest al messaggio di offerta scritto da un LLM su GPU AMD (Token Factory).

Selezione (scelta A+C): rischio > 50 e (almeno 2 ordini oppure spesa netta > 500 Â£).
Spesa netta = acquisti meno ordini annullati, prima della data di riferimento (toglie casi come un ordine enorme poi annullato).
Demo: i primi 200 (o N, primo argomento) per prioritÃ  = rischio x spesa netta.
All'LLM vanno dati giÃ  interpretati (variazioni, andamento annuale e recente, ritardo rispetto al solito) e il prodotto
piÃ¹ comprato. L'andamento recente Ã¨ dinamico: ordini negli ultimi 3 x intervallo medio del cliente contro il periodo prima;
restituisce in JSON motivo, offerta, eventuale sconto, spiegazione per il negozio e messaggio al cliente
che cita il prodotto. Le risposte sono salvate man mano in ../data/llm_offers.jsonl (si puÃ² riprendere).
Scrive ../data/lastcall_offers.csv.
Uso: python pipeline.py [N]   (TOKENFACTORY_API_KEY in churn/.env)
"""
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from features import clean, load_transactions

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = DATA / "llm_offers.jsonl"
BASE_URL = "https://developer.amd.com.cn/radeon/api/v1"
MODEL = os.environ.get("LASTCALL_MODEL", "Qwen3.8-27B")
REASONING = "low"
CUTOFF = pd.Timestamp("2011-09-10")
DEMO_SIZE = 200
MIN_INTERVAL = 3.2  # secondi tra due richieste: resta sotto le 20 richieste al minuto della chiave di team

REASONS = ["price", "lost_interest", "seasonal", "bad_experience", "other"]
OFFERS = ["discount", "free_shipping", "loyalty_points", "similar_products"]

SYSTEM = f"""You are a retention assistant for a UK online gift and homeware wholesaler.
You receive a summary of one customer whom a churn model flagged as at risk of not buying again.
The numbers are already interpreted for you: trust the trend labels and percentages.
The recent trend uses a window of 3x the customer's own usual gap, so it shows whether they stopped recently;
it matters more than the yearly trend.
Reply with a JSON object with exactly these keys:
- "reason": the most likely reason for the slowdown, one of {REASONS}
- "offer": the retention offer that fits best, one of {OFFERS}
- "discount_pct": an integer 5-20 if offer is "discount", otherwise 0
- "explanation": one short sentence (max 30 words) for the shop owner: why this reason and this offer
- "message": a short, friendly email to the customer (max 70 words, English, no name placeholders, no invented
  facts). It must mention their most bought product by name. It must contain exactly the chosen offer and no
  other incentive: discount -> the percentage; free_shipping -> free shipping; loyalty_points -> bonus loyalty
  points; similar_products -> a selection of products similar to their favourite, with no discount, free
  shipping or points.
Guidelines: discount if spending per order dropped; free_shipping for small but frequent orders;
loyalty_points for long-standing customers with many orders; similar_products if orders are getting rarer;
bad_experience only if cancellations are high."""

_lock = threading.Lock()
_last = [0.0]


def api_key() -> str:
    for line in (HERE / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("TOKENFACTORY_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("TOKENFACTORY_API_KEY mancante in churn/.env")


def throttle():
    with _lock:
        wait = _last[0] + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()


def ask(state: dict, key: str, tries: int = 6) -> tuple[dict, dict]:
    body = {"model": MODEL, "reasoning_effort": REASONING, "temperature": 0, "max_tokens": 3000,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": json.dumps(state, ensure_ascii=False)}]}
    for i in range(tries):
        throttle()
        req = urllib.request.Request(BASE_URL + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.load(r)
            text = resp["choices"][0]["message"]["content"] or ""
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            return json.loads(text), resp.get("usage", {})
        except urllib.error.HTTPError as e:
            if e.code not in (429, 502, 503) or i == tries - 1:
                raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}")
            time.sleep(float(e.headers.get("Retry-After") or 2 ** i) + random.random())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
            if i == tries - 1:
                raise
            time.sleep(2 ** i + random.random())


def trend(last: float, prev: float) -> str:
    if prev == 0:
        return "new customer this year" if last > 0 else "no orders"
    ch = (last - prev) / prev
    return "falling" if ch < -0.2 else "growing" if ch > 0.2 else "stable"


def pct(new, old):
    if pd.isna(new) or pd.isna(old) or old == 0:
        return "no comparison available"
    return f"{(new - old) / old:+.0%}"


def to_state(r: pd.Series) -> dict:
    usual = r["gap"]
    late = r["days_since_last_purchase"] / usual if usual else None
    w = int(r["recent_window"])
    return {
        "churn_risk": f"{int(r['risk_score'])}/100",
        "days_since_last_purchase": int(r["days_since_last_purchase"]),
        "usual_days_between_purchases": round(float(usual)) if r["n_purchase_days"] > 1 else "unknown (one purchase day)",
        "delay_vs_usual_rhythm": f"{late:.1f}x the usual gap" if r["n_purchase_days"] > 1 else "unknown",
        f"orders_last_{w}_days (3x usual gap)": int(r["orders_recent"]),
        f"orders_in_the_{w}_days_before": int(r["orders_before"]),
        "recent_order_trend": trend(r["orders_recent"], r["orders_before"]),
        "orders_last_12_months": int(r["n_orders_last_12m"]),
        "orders_previous_12_months": int(r["n_orders_prev_12m"]),
        "yearly_order_trend": trend(r["n_orders_last_12m"], r["n_orders_prev_12m"]),
        "orders_total": int(r["n_orders"]),
        "avg_order_value_last_12_months_gbp": None if pd.isna(r["avg_order_value_last_12m"])
        else round(float(r["avg_order_value_last_12m"])),
        "avg_order_value_change_vs_previous_year": pct(r["avg_order_value_last_12m"], r["avg_order_value_prev_12m"]),
        "net_spend_gbp": round(float(r["net_spend"])),
        "cancelled_orders": int(r["n_cancelled"]),
        "cancellations_level": "high" if r["n_cancelled"] >= max(3, 0.3 * r["n_orders"]) else
        "some" if r["n_cancelled"] > 0 else "none",
        "different_products_bought": int(r["n_products"]),
        "bought_in_same_season_last_year": bool(r["bought_same_period_last_year"]),
        "most_bought_product": str(r["favourite_category"]).strip().title(),
        "customer_since_days": int(r["first_purchase_days_ago"]),
        "country": r["country"],
    }


def mentions_product(msg: str, product: str) -> bool:
    words = [w for w in re.findall(r"[a-z]+", product.lower()) if len(w) > 3]
    return product.lower() in msg or sum(w in msg for w in words) >= min(2, len(words))


def valid(a: dict, product: str) -> bool:
    """Formato giusto, messaggio coerente con l'offerta scelta e che cita il prodotto preferito."""
    if not (a.get("reason") in REASONS and a.get("offer") in OFFERS and isinstance(a.get("message"), str)
            and isinstance(a.get("explanation"), str)):
        return False
    msg = a["message"].lower()
    mentions = {"discount": "%" in msg, "free_shipping": "shipping" in msg, "loyalty_points": "point" in msg}
    if a["offer"] == "similar_products":
        coherent = not any(mentions.values())
    else:
        coherent = mentions[a["offer"]] and sum(mentions.values()) == 1
    return coherent and mentions_product(msg, product)


def ask_valid(state: dict, key: str, tries: int = 3) -> tuple[dict, dict, int]:
    """Richiede di nuovo se il messaggio non Ã¨ coerente o non cita il prodotto."""
    for n in range(1, tries + 1):
        answer, usage = ask(state, key)
        if valid(answer, state["most_bought_product"]):
            break
    return answer, usage, n


def load_customers() -> pd.DataFrame:
    s = pd.read_csv(DATA / "customers_scored_rf.csv", dtype={"customer_id": str})
    c = pd.read_csv(DATA / "customers.csv", dtype={"customer_id": str})[["customer_id", "n_orders",
                                                                         "first_purchase_days_ago"]]
    s = s.merge(c, on="customer_id")
    df = load_transactions()
    buys, returns = clean(df)
    returns = returns[returns["InvoiceDate"] < CUTOFF]
    cancelled = (returns["Quantity"] * returns["Price"]).groupby(returns["customer_id"]).sum()  # valori negativi
    s["net_spend"] = (s["total_spend"] + s["customer_id"].map(cancelled).fillna(0)).clip(lower=0)

    # Ritmo personale: intervallo medio tra giorni di acquisto (90 se ha comprato in un solo giorno, come nella regola)
    past = buys[buys["InvoiceDate"] < CUTOFF].assign(day=lambda d: d["InvoiceDate"].dt.normalize())
    days = past.groupby("customer_id")["day"].agg(lambda x: sorted(x.unique()))
    gap = days.map(lambda d: (d[-1] - d[0]).days / (len(d) - 1) if len(d) > 1 else 90.0)
    s["n_purchase_days"] = s["customer_id"].map(days.map(len))
    s["gap"] = s["customer_id"].map(gap)
    # Andamento recente dinamico: ordini negli ultimi 3 x intervallo medio contro lo stesso periodo prima.
    # Finestra tra 30 e 300 giorni, cosÃ¬ anche il periodo prima resta dentro i dati (che iniziano a dic 2009).
    s["recent_window"] = (3 * s["gap"]).clip(30, 300).round()
    inv = past.drop_duplicates("Invoice")[["customer_id", "InvoiceDate"]]
    inv = inv.merge(s[["customer_id", "recent_window"]], on="customer_id")
    age = (CUTOFF - inv["InvoiceDate"]).dt.total_seconds() / 86400
    inv["recent"] = age < inv["recent_window"]
    inv["before"] = (age >= inv["recent_window"]) & (age < 2 * inv["recent_window"])
    counts = inv.groupby("customer_id")[["recent", "before"]].sum()
    s["orders_recent"] = s["customer_id"].map(counts["recent"]).fillna(0).astype(int)
    s["orders_before"] = s["customer_id"].map(counts["before"]).fillna(0).astype(int)
    return s


if __name__ == "__main__":
    t0 = time.perf_counter()
    s = load_customers()
    sel = s[(s["risk_score"] > 50) & ((s["n_orders"] >= 2) | (s["net_spend"] > 500))].copy()
    sel["priority"] = sel["risk_score"] / 100 * sel["net_spend"]
    size = int(sys.argv[1]) if len(sys.argv) > 1 else DEMO_SIZE
    demo = sel.sort_values("priority", ascending=False).head(size)
    print(f"Clienti attivi {len(s)}, selezionati (A+C) {len(sel)}, demo {len(demo)}, modello {MODEL} "
          f"(reasoning {REASONING})")

    done = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("model") == MODEL:
                done[rec["customer_id"]] = rec
    todo = [r for _, r in demo.iterrows() if r["customer_id"] not in done]
    print(f"GiÃ  fatti {len(done)}, da fare {len(todo)}")

    key, errors = api_key(), 0
    with OUT.open("a", encoding="utf-8") as f, ThreadPoolExecutor(6) as pool:
        futures = {pool.submit(ask_valid, to_state(r), key): (r["customer_id"], to_state(r)) for r in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            cid, state = futures[fut]
            try:
                answer, usage, attempts = fut.result()
            except Exception as e:
                errors += 1
                print(f"  errore {cid}: {e}")
                continue
            rec = {"customer_id": cid, "model": MODEL, "state": state, "answer": answer, "attempts": attempts,
                   "valid": valid(answer, state["most_bought_product"]), "usage": usage}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[cid] = rec
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}  {time.perf_counter() - t0:.0f}s")

    rows = []
    for _, r in demo.iterrows():
        rec = done.get(r["customer_id"])
        a = rec["answer"] if rec else {}
        rows.append({"customer_id": r["customer_id"], "risk_score": r["risk_score"], "fascia": r["fascia"],
                     "net_spend": round(r["net_spend"], 2), "n_orders": r["n_orders"],
                     "days_since_last_purchase": r["days_since_last_purchase"], "country": r["country"],
                     "favourite_product": r["favourite_category"],
                     "reason": a.get("reason"), "offer": a.get("offer"), "discount_pct": a.get("discount_pct"),
                     "explanation": a.get("explanation"), "message": a.get("message"),
                     "valid": rec["valid"] if rec else False, "attempts": rec.get("attempts") if rec else None,
                     "churned_reale": r["churned_reale"]})
    res = pd.DataFrame(rows)
    res.to_csv(DATA / "lastcall_offers.csv", index=False)

    tokens = sum((done[c].get("usage") or {}).get("total_tokens", 0) for c in res["customer_id"] if c in done)
    cost = sum((done[c].get("usage") or {}).get("cost", 0) or 0 for c in res["customer_id"] if c in done)
    print(f"\nRisposte: {res['reason'].notna().sum()}/{len(res)}, valide {res['valid'].sum()}, errori {errors}, "
          f"token {tokens}, costo {cost:.5f} $, tempo {time.perf_counter() - t0:.0f}s")
    print(res["offer"].value_counts().to_string())
