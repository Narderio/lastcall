"""risk_model.py con l'etichetta nuova: andato via = nessun acquisto entro la soglia personale
(2 x intervallo medio, tra 30 e 90 giorni), colonna "nuovo" di churn_rule_compare.csv al 2011-09-10.

Uso: python risk_model_v2.py   (dopo features.py e churn_rule_compare.py)
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from risk_model import DATA, SEED, make_features, metrics

CUTOFF = "2011-09-10"

if __name__ == "__main__":
    t0 = time.perf_counter()
    c = pd.read_csv(DATA / "customers.csv", dtype={"customer_id": str})
    c = c[c["n_orders_last_12m"] >= 1].reset_index(drop=True)
    rules = pd.read_csv(DATA / "churn_rule_compare.csv", dtype={"customer_id": str})
    rules = rules[rules["data_taglio"] == CUTOFF].set_index("customer_id")
    c["churned_old"] = c["churned"]
    c["churned"] = rules["nuovo"].reindex(c["customer_id"]).to_numpy()
    c["soglia"] = rules["soglia"].reindex(c["customer_id"]).to_numpy()
    assert c["churned"].notna().all()

    X, y = make_features(c), c["churned"].astype(int)
    print(f"Clienti attivi: {len(c)}  churned nuova: {y.mean():.1%}  (vecchia {c['churned_old'].mean():.1%})")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    _, X_te_old = train_test_split(X, test_size=0.25, stratify=c["churned_old"], random_state=SEED)
    same = len(set(c.loc[X_te.index, "customer_id"]) & set(c.loc[X_te_old.index, "customer_id"]))
    changed = int((c.loc[X_te.index, "churned"] != c.loc[X_te.index, "churned_old"]).sum())
    print(f"Test: {len(X_te)} clienti, {same} uguali al test di ieri; etichetta diversa da ieri: {changed}")

    models = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        results[name] = metrics(y_te, m.predict_proba(X_te)[:, 1])
    te = c.loc[X_te.index]
    results["Regola giorni>soglia personale"] = metrics(y_te, (te["days_since_last_purchase"] > te["soglia"]).astype(float))
    results["Regola giorni>90"] = metrics(y_te, (te["days_since_last_purchase"] > 90).astype(float))
    print("\nTest set (25%), etichetta nuova:")
    print(pd.DataFrame(results).T.round(3).to_string())

    best = max(models, key=lambda n: results[n]["AUC"])
    model = models[best]
    print(f"\nModello migliore: {best}")
    imp = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=SEED)
    top5 = pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean, "std": imp.importances_std})
    print("\nTop 5 feature (permutation importance, calo di AUC):")
    print(top5.sort_values("importance", ascending=False).head(5).round(4).to_string(index=False))

    oof = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    print(f"\nOut-of-fold 5 fold: AUC {roc_auc_score(y, oof):.3f}  Brier {brier_score_loss(y, oof):.3f}")
    c["risk_score"] = np.round(oof * 100).astype(int)
    c.sort_values("risk_score", ascending=False).to_csv(DATA / "customers_scored_v2.csv", index=False)
    print(f"Tempo: {time.perf_counter() - t0:.1f}s  salvato in {DATA / 'customers_scored_v2.csv'}")
