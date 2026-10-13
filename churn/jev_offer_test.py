"""Prova della nuova domanda best_offer di Jev su 100 clienti del test set.

Nello state non c'è favourite_category; c'è la probabilità di churn già stimata da Jev (jev_answers.jsonl).
Uso: python jev_offer_test.py   (OPENROUTER_API_KEY nell'ambiente)
"""
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from sklearn.model_selection import train_test_split

from jev_batch import clean
from jev_risk import MODEL, URL
from risk_model import DATA, SEED, make_features

N = 100
OUT = DATA / "jev_offer_test.jsonl"

OFFER_QUESTION = {
    "best_offer": {
        "type": "choice",
        "instructions": "Which retention offer fits this customer best? Base the choice on the observable numbers, "
                        "not on the product names.",
        "criteria": {
            "discount": "Average order value in the last 12 months is clearly lower than in the previous 12 months "
                        "(spending per order dropped) and the churn probability is not low.",
            "free_shipping": "Small average order value but frequent orders: shipping cost weighs on each order.",
            "loyalty_points": "Long-standing customer (first purchase long ago) with many orders overall, "
                              "worth rewarding to keep them engaged.",
            "similar_products": "Orders are becoming less frequent (fewer orders in the last 12 months than before, "
                                "or many days since the last purchase): suggest products similar to past purchases.",
            "no_offer": "Churn probability is low and buying behaviour is stable or growing: no offer needed.",
        },
    },
}


def to_state(row, churn_prob):
    keys = ("days_since_last_purchase", "n_orders_last_12m", "n_orders_prev_12m", "avg_order_value_last_12m",
            "avg_order_value_prev_12m", "total_spend", "n_returns_last_12m", "country", "first_purchase_days_ago")
    return {k: clean(row[k]) for k in keys} | {"currency": "GBP", "churn_probability": round(churn_prob, 2)}


def ask_offer(state):
    req = urllib.request.Request(
        URL,
        data=json.dumps({"model": MODEL, "state": state, "questions": OFFER_QUESTION}).encode(),
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


if __name__ == "__main__":
    t0 = time.perf_counter()
    c = pd.read_csv(DATA / "customers.csv", dtype={"customer_id": str})
    c = c[c["n_orders_last_12m"] >= 1].reset_index(drop=True)
    X = make_features(c)
    _, X_te = train_test_split(X, test_size=0.25, stratify=c["churned"], random_state=SEED)
    test = c.loc[X_te.index].head(N)

    prev = {}
    for line in (DATA / "jev_answers.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        prev[rec["customer_id"]] = rec["answers"]

    rows = [r for _, r in test.iterrows()]
    states = [to_state(r, prev[r["customer_id"]]["will_churn"]["noul"]) for r in rows]
    with ThreadPoolExecutor(8) as pool:
        outs = list(pool.map(ask_offer, states))

    with OUT.open("w", encoding="utf-8") as f:
        for r, s, o in zip(rows, states, outs):
            f.write(json.dumps({"customer_id": r["customer_id"], "state": s, "answers": o["answers"],
                                "cost": o.get("usage", {}).get("cost")}) + "\n")

    offers = pd.Series([o["answers"]["best_offer"]["choice"] for o in outs])
    print(f"Distribuzione offerte su {N} clienti:")
    print(offers.value_counts().to_string())

    print("\nEsempi (uno per offerta, poi altri):")
    shown, picked = set(), []
    for i, o in enumerate(outs):
        ch = o["answers"]["best_offer"]["choice"]
        if ch not in shown:
            shown.add(ch)
            picked.append(i)
    picked += [i for i in range(N) if i not in picked][: max(0, 5 - len(picked))]
    for i in picked[:5]:
        r, s, o = rows[i], states[i], outs[i]["answers"]["best_offer"]
        print(f"- {r['customer_id']}: giorni {s['days_since_last_purchase']}, ordini {s['n_orders_last_12m']}/"
              f"{s['n_orders_prev_12m']}, AOV {s['avg_order_value_last_12m']}/{s['avg_order_value_prev_12m']}, "
              f"spesa {s['total_spend']}, resi {s['n_returns_last_12m']}, primo acquisto {s['first_purchase_days_ago']}g, "
              f"p_churn {s['churn_probability']}, churned reale {r['churned']} | motivo "
              f"{prev[r['customer_id']]['main_reason']['choice']} | offerta {o['choice']} {o['probabilities']}")

    cost = sum(o.get("usage", {}).get("cost") or 0 for o in outs)
    print(f"\nCosto: {cost:.4f} $  tempo: {time.perf_counter() - t0:.1f}s")
