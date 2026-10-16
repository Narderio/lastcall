"""Hyperparameter tuning del Random Forest con le 20 feature di model_zoo_v2.py (regola B).

RandomizedSearchCV, 60 combinazioni, 5-fold CV solo sui clienti di train, scoring AUC.
Valutazione finale una sola volta sui 1.080 clienti di test, confronto con il Random Forest attuale (0,760).
Non salva il modello.  Uso: python rf_tuning.py
"""
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from features import clean, load_transactions
from model_zoo import CV
from model_zoo_v2 import CUTOFF, six_features
from risk_model import DATA, SEED, make_features, metrics

warnings.filterwarnings("ignore")

PARAMS = {
    "n_estimators": randint(300, 1001),
    "max_depth": [None] + list(range(6, 21)),
    "min_samples_leaf": randint(10, 81),
    "max_features": [*uniform(0.2, 0.6).rvs(40, random_state=SEED).round(2), "sqrt"],
    "max_samples": uniform(0.5, 0.5),
    "class_weight": [None, "balanced"],
    "criterion": ["gini", "entropy"],
}
CURRENT = {"n_estimators": 400, "max_features": 0.5, "min_samples_leaf": 20}  # da model_zoo.py


def bins(s: pd.Series, edges, labels):
    return pd.cut(s.astype(float), edges, labels=labels, include_lowest=True)


if __name__ == "__main__":
    t0 = time.perf_counter()
    c = pd.read_csv(DATA / "customers.csv", dtype={"customer_id": str})
    c = c[c["n_orders_last_12m"] >= 1].reset_index(drop=True)
    rules = pd.read_csv(DATA / "churn_rule_compare.csv", dtype={"customer_id": str})
    rules = rules[rules["data_taglio"] == str(CUTOFF.date())].set_index("customer_id")
    y = pd.Series(rules["nuovo"].reindex(c["customer_id"]).astype(int).to_numpy(), index=c.index)
    buys, returns = clean(load_transactions())
    X = pd.concat([make_features(c), six_features(buys, returns, c["customer_id"])], axis=1)
    _, X_te_idx = train_test_split(X, test_size=0.25, stratify=c["churned"], random_state=SEED)
    te = X_te_idx.index
    tr = X.index.difference(te)
    X_tr, X_te, y_tr, y_te = X.loc[tr], X.loc[te], y.loc[tr], y.loc[te]
    print(f"Train {len(tr)}, test {len(te)}, feature {X.shape[1]}")

    search = RandomizedSearchCV(RandomForestClassifier(random_state=SEED, n_jobs=1), PARAMS, n_iter=60, cv=CV,
                                scoring="roc_auc", n_jobs=-1, random_state=SEED, refit=True)
    t = time.perf_counter()
    search.fit(X_tr, y_tr)
    print(f"Ricerca: {time.perf_counter() - t:.0f}s")

    r = pd.DataFrame(search.cv_results_)
    cols = {f"param_{k}": k for k in PARAMS}
    view = r[list(cols) + ["mean_test_score", "std_test_score", "rank_test_score"]].rename(columns=cols)
    pd.set_option("display.width", 250)
    print("\nLe 5 combinazioni migliori (AUC media in CV ± dev. std):")
    print(view.sort_values("rank_test_score").head(5).round(4).to_string(index=False))

    print(f"\nAUC in CV: migliore {view['mean_test_score'].max():.4f}, peggiore {view['mean_test_score'].min():.4f}, "
          f"mediana {view['mean_test_score'].median():.4f}")
    print("\nQuali parametri contano (AUC media in CV per gruppo di valori, e quante combinazioni):")
    groups = {
        "class_weight": view["class_weight"].map(lambda v: "balanced" if v == "balanced" else "None"),
        "criterion": view["criterion"],
        "max_depth": view["max_depth"].map(lambda d: "None" if d is None or pd.isna(d) else ("6-10" if d <= 10 else "11-20")),
        "max_features": view["max_features"].map(lambda f: "sqrt" if f == "sqrt" else
                                                  ("0.2-0.4" if float(f) < 0.4 else "0.4-0.6" if float(f) < 0.6 else "0.6-0.8")),
        "min_samples_leaf": bins(view["min_samples_leaf"], [10, 25, 45, 80], ["10-25", "26-45", "46-80"]),
        "max_samples": bins(view["max_samples"], [0.5, 0.67, 0.84, 1.0], ["50-67%", "67-84%", "84-100%"]),
        "n_estimators": bins(view["n_estimators"], [300, 550, 800, 1000], ["300-550", "551-800", "801-1000"]),
    }
    spread = {}
    for name, g in groups.items():
        agg = view.groupby(g, observed=True)["mean_test_score"].agg(["mean", "count"])
        spread[name] = agg["mean"].max() - agg["mean"].min()
        print(f"  {name}: " + ", ".join(f"{k} {v['mean']:.4f} (n={int(v['count'])})" for k, v in agg.iterrows()))
    print("\nDifferenza tra il gruppo migliore e il peggiore, per parametro:")
    for k, v in sorted(spread.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v:.4f}")

    best = search.best_estimator_
    p_new = best.predict_proba(X_te)[:, 1]
    cur = RandomForestClassifier(**CURRENT, n_jobs=-1, random_state=SEED).fit(X_tr, y_tr)
    p_cur = cur.predict_proba(X_te)[:, 1]
    res = pd.DataFrame({
        "Random Forest attuale": {**metrics(y_te, p_cur), "F1 churn": f1_score(y_te, p_cur >= 0.5)},
        "Random Forest dopo il tuning": {**metrics(y_te, p_new), "F1 churn": f1_score(y_te, p_new >= 0.5)},
    }).T
    print(f"\nParametri migliori: {search.best_params_}")
    print(f"AUC in CV del migliore: {search.best_score_:.4f}")
    print("\nTest (1.080 clienti):")
    print(res.round(3).to_string())
    print(f"\nTempo totale: {time.perf_counter() - t0:.0f}s")
