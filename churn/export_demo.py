"""Esporta i dati della demo LastCall in ../demo/data.json.

Fonti: ../data/customers_scored_rf.csv (punteggi del Random Forest per tutti i clienti attivi),
../data/lastcall_offers.csv (offerte scritte dall'LLM per i primi 200) e ../data/customers.csv
(solo per contare i clienti totali). La demo non mostra dettagli tecnici: niente nomi di modelli o metriche.
Nel JSON solo ID anonimi, numeri aggregati del cliente e testi generati: nessun indirizzo email,
nome o altro dato personale (il dataset non ne contiene; non aggiungiamo nient'altro).
Uso: python export_demo.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "demo" / "data.json"
REFERENCE_DATE = "2011-09-10"


def num(v, nd=2):
    return None if pd.isna(v) else round(float(v), nd)


def usual_gap(r):
    """Intervallo medio tra acquisti, ricavato da giorni / rapporto. features usa 90 per chi ha comprato
    in un solo giorno: in quel caso (90 con pochissimi ordini) il ritmo non è noto e restituiamo None."""
    ratio = r["recency_over_gap"]
    if not 0 < ratio < 20:
        return None
    gap = round(r["days_since_last_purchase"] / ratio)
    if gap == 90 and r["n_orders_last_12m"] + r["n_orders_prev_12m"] <= 3:
        return None
    return gap


def pct_change(new, old):
    if pd.isna(new) or pd.isna(old) or old == 0:
        return None
    return round((new - old) / old * 100)


if __name__ == "__main__":
    scored = pd.read_csv(DATA / "customers_scored_rf.csv", dtype={"customer_id": str})
    # clienti totali = tutti quelli con almeno un acquisto prima della data di riferimento
    total_customers = len(pd.read_csv(DATA / "customers.csv", usecols=["customer_id"]))
    offers = pd.read_csv(DATA / "lastcall_offers.csv", dtype={"customer_id": str})
    s = scored.set_index("customer_id")

    bands = []
    for name, lo, hi in (("Low", 0, 39), ("Medium", 40, 69), ("High", 70, 100)):
        b = scored[scored["risk_score"].between(lo, hi)]
        bands.append({"name": name, "range": f"{lo}-{hi}", "customers": int(len(b)),
                      "left": int(b["churned_reale"].sum()), "stayed": int(len(b) - b["churned_reale"].sum())})

    customers = []
    for rank, o in enumerate(offers.itertuples(index=False), 1):
        r = s.loc[o.customer_id]
        ratio = r["recency_over_gap"]
        # intervallo medio ricavato dal rapporto giorni / intervallo (90 per chi ha comprato in un solo giorno)
        gap = usual_gap(r)
        customers.append({
            "id": o.customer_id,
            "rank": rank,
            "risk": int(o.risk_score),
            "band": str(o.fascia),
            "country": o.country,
            "net_spend": num(o.net_spend),
            "orders": int(o.n_orders),
            "orders_last_12m": int(r["n_orders_last_12m"]),
            "orders_prev_12m": int(r["n_orders_prev_12m"]),
            "days_silent": int(o.days_since_last_purchase),
            "usual_gap": gap,
            "delay_ratio": num(ratio, 1),
            "aov_last_12m": num(r["avg_order_value_last_12m"]),
            "aov_change_pct": pct_change(r["avg_order_value_last_12m"], r["avg_order_value_prev_12m"]),
            "cancelled": int(r["n_cancelled"]),
            "products": int(r["n_products"]),
            "same_season_last_year": bool(r["bought_same_period_last_year"]),
            "favourite": str(o.favourite_product).strip().title(),
            "reason": o.reason,
            "offer": o.offer,
            "discount_pct": int(o.discount_pct) if not pd.isna(o.discount_pct) else 0,
            "explanation": o.explanation,
            "message": o.message,
            "checks_ok": bool(o.valid),
        })

    # Clienti attivi senza un'offerta già scritta: la pagina Customers li mostra con le loro statistiche
    with_offer = set(offers["customer_id"])
    pool = []
    for cid, r in s.iterrows():
        if cid in with_offer:
            continue
        ratio = r["recency_over_gap"]
        pool.append({
            "id": cid,
            "risk": int(r["risk_score"]),
            "country": r["country"],
            "total_spend": num(r["total_spend"], 0),
            "orders_last_12m": int(r["n_orders_last_12m"]),
            "orders_prev_12m": int(r["n_orders_prev_12m"]),
            "days_silent": int(r["days_since_last_purchase"]),
            "usual_gap": usual_gap(r),
            "delay_ratio": num(ratio, 1),
            "aov_last_12m": num(r["avg_order_value_last_12m"], 0),
            "aov_change_pct": pct_change(r["avg_order_value_last_12m"], r["avg_order_value_prev_12m"]),
            "cancelled": int(r["n_cancelled"]),
            "products": int(r["n_products"]),
            "same_season_last_year": bool(r["bought_same_period_last_year"]),
            "favourite": str(r["favourite_category"]).strip().title(),
        })
    pool.sort(key=lambda c: -c["risk"])

    data = {
        "reference_date": REFERENCE_DATE,
        "total_customers": int(total_customers),
        "active_customers": int(len(scored)),
        "worth_saving": 1653,  # selezione A+C di pipeline.py: rischio > 50 e (≥ 2 ordini o spesa netta > 500 £)
        "bands": bands,
        "customers": customers,
        "pool": pool,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Clienti attivi {len(scored)}, offerte {len(customers)}, senza offerta {len(pool)} → {OUT} "
          f"({OUT.stat().st_size // 1024} KB)")
