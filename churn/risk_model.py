"""Modello di rischio churn sui clienti attivi: confronto LogisticRegression vs HistGradientBoosting.

Uso: python risk_model.py   (dopo features.py; legge ../data/customers.csv)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parent.parent / "data"
SEED = 42


def make_features(c: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=c.index)
    for col in ("days_since_last_purchase", "n_orders", "n_orders_last_12m", "n_orders_prev_12m",
                "n_returns_last_12m", "first_purchase_days_ago"):
        X[col] = c[col]
    X["log_total_spend"] = np.log1p(c["total_spend"])
    for col in ("avg_order_value", "avg_order_value_last_12m", "avg_order_value_prev_12m"):
        X[f"log_{col}"] = np.log1p(c[col].fillna(0))
    X["no_orders_prev_12m"] = c["avg_order_value_prev_12m"].isna().astype(int)
    X["order_ratio"] = c["n_orders_last_12m"] / (c["n_orders_prev_12m"] + 1)
    prev = c["avg_order_value_prev_12m"]
    X["aov_change"] = ((c["avg_order_value_last_12m"] - prev) / prev).fillna(0).clip(-1, 10)
    X["is_uk"] = (c["country"] == "United Kingdom").astype(int)
    return X


def metrics(y, p, threshold=0.5) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        "AUC": roc_auc_score(y, p),
        "accuratezza": accuracy_score(y, pred),
        "precisione churn": precision_score(y, pred),
        "recall churn": recall_score(y, pred),
        "Brier": brier_score_loss(y, p),
    }


if __name__ == "__main__":
    t0 = time.perf_counter()
    c = pd.read_csv(DATA / "customers.csv", dtype={"customer_id": str})
    c = c[c["n_orders_last_12m"] >= 1].reset_index(drop=True)
    X, y = make_features(c), c["churned"]
    print(f"Clienti attivi: {len(c)}  churned: {y.mean():.1%}  feature: {X.shape[1]}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    models = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
    }
    results = {}
    for name, m in models.items():
        t = time.perf_counter()
        m.fit(X_tr, y_tr)
        results[name] = metrics(y_te, m.predict_proba(X_te)[:, 1])
        results[name]["tempo s"] = time.perf_counter() - t
    rule = (X_te["days_since_last_purchase"] > 90).astype(float)
    results["Regola giorni>90"] = metrics(y_te, rule)

    print("\nTest set (25%):")
    print(pd.DataFrame(results).T.round(3).to_string())

    best = max(models, key=lambda n: results[n]["AUC"])
    print(f"\nModello migliore: {best}")
    model = models[best]

    imp = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=SEED)
    importance = pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean, "std": imp.importances_std})
    top5 = importance.sort_values("importance", ascending=False).head(5)
    top5.round(4).to_csv(DATA / "feature_importance.csv", index=False)
    print("\nTop 5 feature (permutation importance, calo di AUC):")
    print(top5.round(4).to_string(index=False))

    t = time.perf_counter()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    print(f"\nOut-of-fold 5 fold su tutti gli attivi: AUC {roc_auc_score(y, oof):.3f}  Brier {brier_score_loss(y, oof):.3f}"
          f"  ({time.perf_counter() - t:.1f}s)")

    c["risk_score"] = np.round(oof * 100).astype(int)
    scored = c.sort_values("risk_score", ascending=False)
    scored.to_csv(DATA / "customers_scored.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\nClienti a rischio più alto:")
    print(scored.head(5).to_string(index=False))
    print(f"\nTempo totale: {time.perf_counter() - t0:.1f}s  salvato in {DATA / 'customers_scored.csv'}")
