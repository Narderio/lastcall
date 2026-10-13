"""Classifica dei 16 modelli di model_zoo.py prima e dopo 6 feature nuove (regola B, stesso test di 1.080 clienti).

Feature nuove, solo con transazioni prima del 2011-09-10:
 1) recency_over_gap: giorni dall'ultimo acquisto / intervallo medio tra giorni di acquisto
    (chi ha un solo giorno di acquisto: intervallo = 90, come per la soglia)
 2) orders_3m_ratio: ordini negli ultimi 90 giorni / (ordini nei 90 giorni prima + 1)
 3) aov_3m_over_alltime: spesa media per ordine ultimi 90 giorni / spesa media di sempre (0 se nessun ordine)
 4) n_cancelled: numero di fatture annullate (Invoice che inizia con C)
 5) n_products: numero di prodotti diversi comprati
 6) bought_same_period_last_year: acquisto tra il 2010-09-10 e il 2010-12-09 (1/0)
Scrive ../data/model_zoo_v2_results.csv.  Uso: python model_zoo_v2.py
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

from features import clean, load_transactions
from model_zoo import CV, candidates
from risk_model import DATA, SEED, make_features, metrics

warnings.filterwarnings("ignore")
CUTOFF = pd.Timestamp("2011-09-10")
DAY = pd.Timedelta(days=1)


def six_features(buys: pd.DataFrame, returns: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    past = buys[buys["InvoiceDate"] < CUTOFF].copy()
    past["day"] = past["InvoiceDate"].dt.normalize()
    g = past.groupby("customer_id")
    gap = g["day"].apply(lambda s: np.diff(np.sort(s.unique())).astype("timedelta64[D]").astype(float).mean()
                         if s.nunique() > 1 else 90.0)
    recency = (CUTOFF - g["day"].max()).dt.days

    def orders(start, end):
        w = past[(past["InvoiceDate"] >= CUTOFF - start * DAY) & (past["InvoiceDate"] < CUTOFF - end * DAY)]
        return w.groupby("customer_id")["Invoice"].nunique(), w.groupby("customer_id")["amount"].sum()

    o3, s3 = orders(90, 0)
    o6, _ = orders(180, 90)
    aov_all = g["amount"].sum() / g["Invoice"].nunique()
    ly = past[(past["InvoiceDate"] >= CUTOFF - 365 * DAY) & (past["InvoiceDate"] < CUTOFF - 275 * DAY)]
    canc = returns[returns["InvoiceDate"] < CUTOFF].groupby("customer_id")["Invoice"].nunique()

    f = pd.DataFrame(index=ids.to_numpy())
    f["recency_over_gap"] = (recency / gap).reindex(f.index).clip(upper=20)
    f["orders_3m_ratio"] = o3.reindex(f.index, fill_value=0) / (o6.reindex(f.index, fill_value=0) + 1)
    f["aov_3m_over_alltime"] = ((s3 / o3) / aov_all).reindex(f.index).fillna(0).clip(upper=10)
    f["n_cancelled"] = canc.reindex(f.index, fill_value=0)
    f["n_products"] = g["StockCode"].nunique().reindex(f.index)
    f["bought_same_period_last_year"] = f.index.isin(ly["customer_id"].unique()).astype(int)
    return f.reset_index(drop=True)


def ranking(X, y, tr, te):
    X_tr, X_te, y_tr, y_te = X.loc[tr], X.loc[te], y.loc[tr], y.loc[te]
    preds, fitted, cv_scores, secs = {}, {}, {}, {}
    for name, model in candidates().items():
        t = time.perf_counter()
        model.fit(X_tr, y_tr)
        cv_scores[name] = model.best_score_ if isinstance(model, GridSearchCV) else \
            cross_val_score(model, X_tr, y_tr, cv=CV, scoring="roc_auc").mean()
        fitted[name] = model.best_estimator_ if isinstance(model, GridSearchCV) else model
        preds[name], secs[name] = model.predict_proba(X_te)[:, 1], time.perf_counter() - t
    ranked = sorted(cv_scores, key=cv_scores.get, reverse=True)
    ens = {f"Voting ({', '.join(ranked[:3])})": VotingClassifier([(n, fitted[n]) for n in ranked[:3]], voting="soft"),
           f"Stacking ({', '.join(ranked[:5])} → LR)": StackingClassifier(
               [(n, fitted[n]) for n in ranked[:5]], final_estimator=LogisticRegression(max_iter=2000), cv=CV, n_jobs=-1)}
    for name, model in ens.items():
        t = time.perf_counter()
        cv_scores[name] = cross_val_score(model, X_tr, y_tr, cv=CV, scoring="roc_auc", n_jobs=-1).mean()
        model.fit(X_tr, y_tr)
        fitted[name] = model
        preds[name], secs[name] = model.predict_proba(X_te)[:, 1], time.perf_counter() - t
    rows = [{"modello": n, **metrics(y_te, p), "F1 churn": f1_score(y_te, (p >= 0.5).astype(int)),
             "AUC cv train": cv_scores[n], "tempo s": secs[n]} for n, p in preds.items()]
    return pd.DataFrame(rows).sort_values("AUC", ascending=False), fitted


def short(name):
    return name.split(" (")[0]


if __name__ == "__main__":
    t0 = time.perf_counter()
    c = pd.read_csv(DATA / "customers.csv", dtype={"customer_id": str})
    c = c[c["n_orders_last_12m"] >= 1].reset_index(drop=True)
    rules = pd.read_csv(DATA / "churn_rule_compare.csv", dtype={"customer_id": str})
    rules = rules[rules["data_taglio"] == str(CUTOFF.date())].set_index("customer_id")
    y = pd.Series(rules["nuovo"].reindex(c["customer_id"]).astype(int).to_numpy(), index=c.index)

    base = make_features(c)
    buys, returns = clean(load_transactions())
    six = six_features(buys, returns, c["customer_id"])
    full = pd.concat([base, six], axis=1)
    print("Feature nuove (descrizione):")
    print(six.describe().T[["mean", "50%", "min", "max"]].round(2).to_string())

    _, X_te_idx = train_test_split(base, test_size=0.25, stratify=c["churned"], random_state=SEED)
    te = X_te_idx.index
    tr = base.index.difference(te)

    before, _ = ranking(base, y, tr, te)
    after, fitted = ranking(full, y, tr, te)
    before["fase"], after["fase"] = "prima (14 feature)", "dopo (20 feature)"
    pd.concat([before, after]).round(4).to_csv(DATA / "model_zoo_v2_results.csv", index=False)

    cmp = pd.DataFrame({
        "AUC prima": before.assign(m=before["modello"].map(short)).set_index("m")["AUC"],
        "AUC dopo": after.assign(m=after["modello"].map(short)).set_index("m")["AUC"],
        "Brier prima": before.assign(m=before["modello"].map(short)).set_index("m")["Brier"],
        "Brier dopo": after.assign(m=after["modello"].map(short)).set_index("m")["Brier"],
    })
    cmp["delta AUC"] = cmp["AUC dopo"] - cmp["AUC prima"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print("\nPrima / dopo (AUC e Brier sul test di 1.080 clienti), ordinato per AUC dopo:")
    print(cmp.sort_values("AUC dopo", ascending=False).round(3).to_string())
    print("\nClassifica completa dopo:")
    print(after.drop(columns="fase").round(3).to_string(index=False))

    best = after.iloc[0]["modello"]
    imp = permutation_importance(fitted[best], full.loc[te], y.loc[te], scoring="roc_auc", n_repeats=10,
                                 random_state=SEED)
    s = pd.DataFrame({"importanza": imp.importances_mean, "std": imp.importances_std}, index=full.columns)
    s["nuova"] = s.index.isin(six.columns)
    print(f"\nFeature più importanti per il migliore dopo ({best}), calo di AUC:")
    print(s.sort_values("importanza", ascending=False).head(10).round(4).to_string())
    print(f"\nTempo totale: {time.perf_counter() - t0:.0f}s")
