"""Random Forest finale: punteggio di rischio 0-100 per tutti i clienti attivi al 2011-09-10.

20 feature (model_zoo_v2.py), regola B. Parametri: quelli attuali (il tuning non ha migliorato il test).
- risk_score: predizioni out-of-fold a 5 fold, così ogni cliente è valutato da un modello che non l'ha visto.
- controllo sui soliti 1.080 clienti di test (modello addestrato solo sul train) e feature più importanti.
- modello finale addestrato su tutti i clienti attivi, salvato in ../data/models/rf_final.joblib.
Scrive ../data/customers_scored_rf.csv.  Uso: python rf_final.py
"""
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split

from features import clean, load_transactions
from model_zoo import CV
from model_zoo_v2 import CUTOFF, six_features
from risk_model import DATA, SEED, make_features, metrics

PARAMS = {"n_estimators": 400, "max_features": 0.5, "min_samples_leaf": 20, "random_state": SEED, "n_jobs": -1}


if __name__ == "__main__":
    t0 = time.perf_counter()
    c = pd.read_csv(DATA / "customers.csv", dtype={"customer_id": str})
    c = c[c["n_orders_last_12m"] >= 1].reset_index(drop=True)
    rules = pd.read_csv(DATA / "churn_rule_compare.csv", dtype={"customer_id": str})
    rules = rules[rules["data_taglio"] == str(CUTOFF.date())].set_index("customer_id")
    y = pd.Series(rules["nuovo"].reindex(c["customer_id"]).astype(int).to_numpy(), index=c.index)
    buys, returns = clean(load_transactions())
    six = six_features(buys, returns, c["customer_id"])
    X = pd.concat([make_features(c), six], axis=1)
    print(f"Clienti attivi: {len(c)}, feature: {X.shape[1]}, andati via (regola B): {y.mean():.1%}")

    # Controllo sul test di sempre
    _, X_te_idx = train_test_split(X, test_size=0.25, stratify=c["churned"], random_state=SEED)
    te = X_te_idx.index
    tr = X.index.difference(te)
    check = RandomForestClassifier(**PARAMS).fit(X.loc[tr], y.loc[tr])
    p = check.predict_proba(X.loc[te])[:, 1]
    m = metrics(y.loc[te], p)
    print("\nTest (1.080 clienti): " + ", ".join(f"{k} {v:.3f}" for k, v in m.items())
          + f", F1 {f1_score(y.loc[te], p >= 0.5):.3f}")
    imp = permutation_importance(check, X.loc[te], y.loc[te], scoring="roc_auc", n_repeats=10, random_state=SEED)
    top = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False).head(8)
    print("\nFeature più importanti (calo di AUC sul test):")
    print(top.round(4).to_string())

    # Punteggio out-of-fold per tutti
    oof = cross_val_predict(RandomForestClassifier(**PARAMS), X, y, cv=CV, method="predict_proba")[:, 1]
    print(f"\nOut-of-fold su tutti i {len(c)} clienti: AUC {roc_auc_score(y, oof):.3f}, Brier {brier_score_loss(y, oof):.3f}")

    out = c[["customer_id", "country", "days_since_last_purchase", "n_orders_last_12m", "n_orders_prev_12m",
             "total_spend", "avg_order_value_last_12m", "avg_order_value_prev_12m", "n_returns_last_12m",
             "favourite_category"]].copy()
    out["soglia_giorni"] = rules["soglia"].reindex(c["customer_id"]).to_numpy()
    out = pd.concat([out, six.round(2)], axis=1)
    out["risk_score"] = np.round(oof * 100).astype(int)
    out["fascia"] = pd.cut(out["risk_score"], [-1, 39, 69, 100], labels=["basso", "medio", "alto"])
    out["churned_reale"] = y.to_numpy()
    out = out.sort_values("risk_score", ascending=False)
    out.to_csv(DATA / "customers_scored_rf.csv", index=False)

    print("\nFasce di rischio (punteggio 0-39 basso, 40-69 medio, 70-100 alto):")
    f = out.groupby("fascia", observed=True).agg(clienti=("customer_id", "size"),
                                                  andati_via_davvero=("churned_reale", "mean"))
    f["andati_via_davvero"] = (f["andati_via_davvero"] * 100).round(1).astype(str) + "%"
    print(f.to_string())

    final = RandomForestClassifier(**PARAMS).fit(X, y)
    (DATA / "models").mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": list(X.columns), "cutoff": str(CUTOFF.date()), "params": PARAMS},
                DATA / "models" / "rf_final.joblib")

    pd.set_option("display.width", 250)
    print("\nI 5 clienti a rischio più alto:")
    print(out.head(5)[["customer_id", "risk_score", "days_since_last_purchase", "n_orders_last_12m",
                       "n_orders_prev_12m", "total_spend", "recency_over_gap", "soglia_giorni", "country",
                       "churned_reale"]].to_string(index=False))
    print(f"\nSalvati: {DATA / 'customers_scored_rf.csv'}, {DATA / 'models' / 'rf_final.joblib'}"
          f"  tempo {time.perf_counter() - t0:.0f}s")
