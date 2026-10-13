"""Confronto Jev (OpenRouter Decisions) vs regressione logistica sullo stesso test set di risk_model.py.

Uso: python jev_batch.py   (OPENROUTER_API_KEY nell'ambiente; riprende da ../data/jev_answers.jsonl)
"""
import json
import math
import random
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from jev_risk import decide
from risk_model import DATA, SEED, make_features

OUT = DATA / "jev_answers.jsonl"
WORKERS = 8
COST_PER_CALL = 3.2e-5  # misurato con jev_risk.py
MAX_BUDGET = 0.50


def clean(v):
    if isinstance(v, float):
        return None if math.isnan(v) else round(v, 2)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def to_state(row: pd.Series) -> dict:
    return {k: clean(row[k]) for k in (
        "days_since_last_purchase", "n_orders_last_12m", "n_orders_prev_12m",
        "avg_order_value_last_12m", "avg_order_value_prev_12m", "total_spend",
        "n_returns_last_12m", "country", "favourite_category", "first_purchase_days_ago")} | {
        "currency": "GBP"}


def ask(state: dict, tries: int = 6, decide_fn=decide) -> dict:
    for i in range(tries):
        try:
            return decide_fn(state)
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500 or i == tries - 1:
                raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
        except (urllib.error.URLError, TimeoutError):
            if i == tries - 1:
                raise
        time.sleep(2 ** i + random.random())


def metrics(y, p) -> dict:
    pred = (p >= 0.5).astype(int)
    return {"AUC": roc_auc_score(y, p), "accuratezza": accuracy_score(y, pred),
            "precisione churn": precision_score(y, pred), "recall churn": recall_score(y, pred),
            "Brier": brier_score_loss(y, p)}


if __name__ == "__main__":
    t0 = time.perf_counter()
    c = pd.read_csv(DATA / "customers.csv", dtype={"customer_id": str})
    c = c[c["n_orders_last_12m"] >= 1].reset_index(drop=True)
    X, y = make_features(c), c["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    test = c.loc[X_te.index]

    done = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["customer_id"]] = rec
    todo = [r for _, r in test.iterrows() if r["customer_id"] not in done]
    estimate = len(todo) * COST_PER_CALL
    print(f"Test set: {len(test)} clienti, già fatti: {len(done)}, da fare: {len(todo)}, costo stimato: {estimate:.3f} $")
    if estimate > MAX_BUDGET:
        raise SystemExit(f"Stima sopra {MAX_BUDGET} $: mi fermo.")

    errors = 0
    t1 = time.perf_counter()
    with OUT.open("a", encoding="utf-8") as f, ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(ask, to_state(r)): r["customer_id"] for r in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            cid = futures[fut]
            try:
                out = fut.result()
            except Exception as e:
                errors += 1
                print(f"  errore {cid}: {e}")
                continue
            rec = {"customer_id": cid, "answers": out["answers"], "cost": out.get("usage", {}).get("cost")}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[cid] = rec
            if n % 100 == 0:
                print(f"  {n}/{len(todo)}  {time.perf_counter() - t1:.0f}s")
    print(f"Chiamate: {len(todo)} in {time.perf_counter() - t1:.1f}s, errori: {errors}")

    test = test[test["customer_id"].isin(done)]
    yt = test["churned"].to_numpy()
    ans = [done[cid]["answers"] for cid in test["customer_id"]]
    noul = np.array([a["will_churn"]["noul"] for a in ans])
    score = np.array([a["risk_level"]["score"] for a in ans])

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
    p_lr = lr.predict_proba(X_te.loc[test.index])[:, 1]

    res = {
        "Jev will_churn": metrics(yt, noul),
        "Jev risk_level (score/3)": metrics(yt, score / 3),
        "LogisticRegression": metrics(yt, p_lr),
        "Regola giorni>90": metrics(yt, (test["days_since_last_purchase"] > 90).astype(float).to_numpy()),
    }
    print(f"\nConfronto su {len(test)} clienti del test set (churned {yt.mean():.1%}):")
    print(pd.DataFrame(res).T.round(3).to_string())
    print(f"\nCorrelazione Jev noul vs LR: {np.corrcoef(noul, p_lr)[0, 1]:.2f}")

    print("\nmain_reason:")
    print(pd.Series([a["main_reason"]["choice"] for a in ans]).value_counts().to_string())

    cost = sum(done[cid]["cost"] or 0 for cid in test["customer_id"])
    print(f"\nCosto totale Jev: {cost:.4f} $  tempo totale: {time.perf_counter() - t0:.1f}s")
