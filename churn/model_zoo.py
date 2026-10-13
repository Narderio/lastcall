"""Confronto di molti modelli (statistici ed ensemble) sull'etichetta nuova (soglia personale, tetto 90).

Test = esattamente i 1.080 clienti di test di risk_model.py (split con l'etichetta vecchia, seed 42).
Train = tutti gli altri clienti attivi al 2011-09-10. Iperparametri scelti con 5-fold CV solo sul train.
Righe di riferimento sugli stessi clienti: Jev (jev_answers.jsonl), regole "giorni > soglia" e "giorni > 90".
Scrive ../data/model_zoo_results.csv.  Uso: python model_zoo.py
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from risk_model import DATA, SEED, make_features, metrics

warnings.filterwarnings("ignore")
CUTOFF = "2011-09-10"
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)


def search(model, grid):
    return GridSearchCV(model, grid, cv=CV, scoring="roc_auc", n_jobs=-1)


def candidates():
    return {
        "LogisticRegression": search(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                                     {"logisticregression__C": [0.01, 0.1, 1, 10]}),
        "GaussianNB": make_pipeline(QuantileTransformer(output_distribution="normal", n_quantiles=200), GaussianNB()),
        "KNN": search(make_pipeline(StandardScaler(), KNeighborsClassifier(weights="distance")),
                      {"kneighborsclassifier__n_neighbors": [15, 30, 50, 100, 200]}),
        "SVM (rbf)": search(make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
                            {"svc__C": [0.3, 1, 3]}),
        "DecisionTree": search(DecisionTreeClassifier(min_samples_leaf=30, random_state=SEED),
                               {"max_depth": [3, 4, 5, 6, 8]}),
        "MLP": search(make_pipeline(StandardScaler(), MLPClassifier(max_iter=1000, early_stopping=True,
                                                                     random_state=SEED)),
                      {"mlpclassifier__hidden_layer_sizes": [(16,), (32, 16)], "mlpclassifier__alpha": [1e-3, 1e-1]}),
        "RandomForest": search(RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=SEED),
                               {"min_samples_leaf": [5, 10, 20], "max_features": ["sqrt", 0.5]}),
        "ExtraTrees": search(ExtraTreesClassifier(n_estimators=400, n_jobs=-1, random_state=SEED),
                             {"min_samples_leaf": [5, 10, 20], "max_features": ["sqrt", 0.5]}),
        "AdaBoost": search(AdaBoostClassifier(random_state=SEED),
                           {"n_estimators": [100, 300], "learning_rate": [0.1, 0.5]}),
        "GradientBoosting": search(GradientBoostingClassifier(subsample=0.8, random_state=SEED),
                                   {"n_estimators": [100, 300], "learning_rate": [0.03, 0.1], "max_depth": [2, 3]}),
        "HistGradientBoosting": search(HistGradientBoostingClassifier(random_state=SEED),
                                       {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [7, 15, 31],
                                        "min_samples_leaf": [20, 50]}),
        "XGBoost": search(XGBClassifier(n_estimators=300, subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                                        n_jobs=1, verbosity=0),
                          {"learning_rate": [0.03, 0.1], "max_depth": [2, 3, 5], "min_child_weight": [1, 10]}),
        "LightGBM": search(LGBMClassifier(n_estimators=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                          random_state=SEED, n_jobs=1, verbose=-1),
                           {"learning_rate": [0.03, 0.1], "num_leaves": [7, 15, 31], "min_child_samples": [20, 50]}),
        "CatBoost": search(CatBoostClassifier(iterations=400, random_seed=SEED, verbose=0, thread_count=1),
                           {"learning_rate": [0.03, 0.1], "depth": [3, 4, 6]}),
    }


def best_estimator(m):
    return m.best_estimator_ if isinstance(m, GridSearchCV) else m


def evaluate(name, y_te, p, cv_auc=np.nan, secs=np.nan):
    r = {"modello": name, **metrics(y_te, p), "F1 churn": f1_score(y_te, (p >= 0.5).astype(int)),
         "AUC cv train": cv_auc, "tempo s": secs}
    return r


if __name__ == "__main__":
    t0 = time.perf_counter()
    c = pd.read_csv(DATA / "customers.csv", dtype={"customer_id": str})
    c = c[c["n_orders_last_12m"] >= 1].reset_index(drop=True)
    rules = pd.read_csv(DATA / "churn_rule_compare.csv", dtype={"customer_id": str})
    rules = rules[rules["data_taglio"] == CUTOFF].set_index("customer_id")
    X = make_features(c)
    y = pd.Series(rules["nuovo"].reindex(c["customer_id"]).astype(int).to_numpy(), index=c.index)
    soglia = rules["soglia"].reindex(c["customer_id"]).to_numpy()

    # Split di ieri (etichetta vecchia), poi etichetta nuova
    _, X_te_idx = train_test_split(X, test_size=0.25, stratify=c["churned"], random_state=SEED)
    te = X_te_idx.index
    tr = X.index.difference(te)
    X_tr, X_te, y_tr, y_te = X.loc[tr], X.loc[te], y.loc[tr], y.loc[te]
    print(f"Train {len(tr)}, test {len(te)} (stessi clienti di ieri), churn nuovo test {y_te.mean():.1%}")

    rows, fitted, cv_scores = [], {}, {}
    for name, model in candidates().items():
        t = time.perf_counter()
        model.fit(X_tr, y_tr)
        if isinstance(model, GridSearchCV):
            cv_auc = model.best_score_
        else:
            cv_auc = cross_val_score(model, X_tr, y_tr, cv=CV, scoring="roc_auc").mean()
        secs = time.perf_counter() - t
        fitted[name], cv_scores[name] = best_estimator(model), cv_auc
        rows.append(evaluate(name, y_te, model.predict_proba(X_te)[:, 1], cv_auc, secs))
        extra = f"  {model.best_params_}" if isinstance(model, GridSearchCV) else ""
        print(f"  {name}: AUC cv {cv_auc:.3f}, test {rows[-1]['AUC']:.3f}  ({secs:.0f}s){extra}")

    ranked = sorted(cv_scores, key=cv_scores.get, reverse=True)
    ensembles = {
        f"Voting ({', '.join(ranked[:3])})": VotingClassifier([(n, fitted[n]) for n in ranked[:3]], voting="soft"),
        f"Stacking ({', '.join(ranked[:5])} → LR)": StackingClassifier(
            [(n, fitted[n]) for n in ranked[:5]], final_estimator=LogisticRegression(max_iter=2000), cv=CV, n_jobs=-1),
    }
    for name, model in ensembles.items():
        t = time.perf_counter()
        cv_auc = cross_val_score(model, X_tr, y_tr, cv=CV, scoring="roc_auc", n_jobs=-1).mean()
        model.fit(X_tr, y_tr)
        secs = time.perf_counter() - t
        fitted[name] = model
        rows.append(evaluate(name, y_te, model.predict_proba(X_te)[:, 1], cv_auc, secs))
        print(f"  {name}: AUC cv {cv_auc:.3f}, test {rows[-1]['AUC']:.3f}  ({secs:.0f}s)")

    # Riferimenti sugli stessi clienti di test
    jev = {}
    for line in (DATA / "jev_answers.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        jev[rec["customer_id"]] = rec["answers"]["will_churn"]["noul"]
    ids = c.loc[te, "customer_id"]
    rows.append(evaluate("Jev (will_churn, senza addestramento)", y_te, ids.map(jev).to_numpy()))
    days = c.loc[te, "days_since_last_purchase"].to_numpy()
    rows.append(evaluate("Regola giorni > soglia personale", y_te, (days > soglia[te]).astype(float)))
    rows.append(evaluate("Regola giorni > 90", y_te, (days > 90).astype(float)))

    res = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    res.round(4).to_csv(DATA / "model_zoo_results.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print("\nRisultati sul test (1.080 clienti di ieri, etichetta nuova), ordinati per AUC:")
    print(res.round(3).to_string(index=False))

    best = next(n for n in res["modello"] if n in fitted)
    imp = permutation_importance(fitted[best], X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=SEED)
    top = pd.DataFrame({"feature": X.columns, "importanza": imp.importances_mean, "std": imp.importances_std})
    print(f"\nMiglior modello: {best}. Top 5 feature (permutation importance, calo di AUC):")
    print(top.sort_values("importanza", ascending=False).head(5).round(4).to_string(index=False))
    print(f"\nTempo totale: {time.perf_counter() - t0:.0f}s  salvato in {DATA / 'model_zoo_results.csv'}")
