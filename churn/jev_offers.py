"""Offerta di Jev (domanda OFFER_QUESTIONS di jev_risk.py) per tutti i clienti del test set.

Usa la p_churn e il motivo già salvati da jev_batch.py. Scrive ../data/jev_offers.csv.
Uso: python jev_offers.py   (OPENROUTER_API_KEY nell'ambiente)
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from sklearn.model_selection import train_test_split

from jev_batch import ask, to_state
from jev_risk import OFFER_QUESTIONS, decide, offer_state
from risk_model import DATA, SEED, make_features

if __name__ == "__main__":
    t0 = time.perf_counter()
    c = pd.read_csv(DATA / "customers.csv", dtype={"customer_id": str})
    c = c[c["n_orders_last_12m"] >= 1].reset_index(drop=True)
    _, X_te = train_test_split(make_features(c), test_size=0.25, stratify=c["churned"], random_state=SEED)
    test = c.loc[X_te.index]

    prev = {}
    for line in (DATA / "jev_answers.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        prev[rec["customer_id"]] = rec["answers"]
    scores = pd.read_csv(DATA / "customers_scored.csv", dtype={"customer_id": str}).set_index("customer_id")["risk_score"]

    rows = [r for _, r in test.iterrows()]
    p_churn = [prev[r["customer_id"]]["will_churn"]["noul"] for r in rows]
    states = [offer_state(to_state(r), p) for r, p in zip(rows, p_churn)]
    with ThreadPoolExecutor(8) as pool:
        outs = list(pool.map(lambda s: ask(s, decide_fn=lambda st: decide(st, OFFER_QUESTIONS)), states))

    out = pd.DataFrame([{
        "customer_id": r["customer_id"],
        "risk_score": scores[r["customer_id"]],
        "p_churn_jev": p,
        "main_reason": prev[r["customer_id"]]["main_reason"]["choice"],
        "best_offer": o["answers"]["best_offer"]["choice"],
        **{f"p_{k}": v for k, v in o["answers"]["best_offer"]["probabilities"].items()},
        "churned": r["churned"],
    } for r, p, o in zip(rows, p_churn, outs)])
    out.to_csv(DATA / "jev_offers.csv", index=False)

    print(f"Offerte su {len(out)} clienti:")
    print(out["best_offer"].value_counts().to_string())
    print("\nOfferte per motivo:")
    print(pd.crosstab(out["main_reason"], out["best_offer"], margins=True, margins_name="totale").to_string())
    cost = sum(o.get("usage", {}).get("cost") or 0 for o in outs)
    print(f"\nCosto: {cost:.4f} $  tempo: {time.perf_counter() - t0:.1f}s")
