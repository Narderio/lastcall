"""Dati per il fine-tuning di Kev sulla domanda will_churn (formato README di jaredpalmer/kev).

test = gli stessi 1080 clienti di test usati per Jev e regressione logistica (seed 42).
train/dev = gli altri clienti attivi, 85/15 diviso per cliente, con in più le loro schede
al 2011-07-27 e al 2011-06-12. Lo state include snapshot_month (stagione). Nessun cliente di test finisce in train/dev.
Uso: python build_kev_data.py   (dopo features.py)
"""
import copy
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from features import HORIZON, build_features, clean, load_transactions
from jev_batch import to_state
from jev_risk import QUESTIONS
from risk_model import DATA, SEED, make_features

OUT = DATA / "kev"
EXTRA_CUTOFFS = ("2011-07-27", "2011-06-12")  # oltre al CUTOFF principale; marzo escluso, storia troppo corta


def active(c: pd.DataFrame) -> pd.DataFrame:
    return c[c["n_orders_last_12m"] >= 1].reset_index(drop=True)


def record(row: pd.Series) -> dict:
    q = copy.deepcopy(QUESTIONS["will_churn"])
    q["label"] = bool(row["churned"])
    state = to_state(row) | {"snapshot_month": pd.Timestamp(row["cutoff"]).month_name()}
    return {"state": state, "questions": {"will_churn": q}}


def write(path, frame):
    with path.open("w", encoding="utf-8") as f:
        for _, row in frame.iterrows():
            f.write(json.dumps(record(row), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main = active(pd.read_csv(DATA / "customers.csv", dtype={"customer_id": str}))
    _, X_te = train_test_split(make_features(main), test_size=0.25, stratify=main["churned"], random_state=SEED)
    df = load_transactions()
    buys, returns = clean(df)
    cutoff = df["InvoiceDate"].max().normalize() - HORIZON
    main = main.assign(cutoff=cutoff.date())
    test = main.loc[X_te.index]
    test_ids = set(test["customer_id"])

    snapshots = [main]
    for day in EXTRA_CUTOFFS:
        c = pd.Timestamp(day)
        snapshots.append(active(build_features(buys, returns, c)).assign(cutoff=c.date()))
        print(f"CUTOFF {c.date()}: {len(snapshots[-1])} clienti attivi")
    pool = pd.concat(snapshots, ignore_index=True)
    pool = pool[~pool["customer_id"].isin(test_ids)]

    # Divisione per cliente, così le schede dello stesso cliente stanno tutte in train o tutte in dev
    per_customer = pool.groupby("customer_id")["churned"].mean().round().astype(int)
    tr_ids, dev_ids = train_test_split(per_customer.index.to_numpy(dtype=object), test_size=0.15,
                                      stratify=per_customer.to_numpy(), random_state=SEED)
    train = pool[pool["customer_id"].isin(set(tr_ids))].sample(frac=1, random_state=SEED)
    dev = pool[pool["customer_id"].isin(set(dev_ids))].sample(frac=1, random_state=SEED)

    assert not (set(train["customer_id"]) | set(dev["customer_id"])) & test_ids
    for name, frame in (("train", train), ("dev", dev), ("test", test)):
        path = OUT / f"{name}.jsonl"
        write(path, frame)
        by_cut = {str(k): (v, round(frame.loc[frame["cutoff"] == k, "churned"].mean(), 3))
                  for k, v in frame["cutoff"].value_counts().sort_index().items()}
        print(f"{name}: {len(frame)} righe, {frame['customer_id'].nunique()} clienti, churn {frame['churned'].mean():.1%}, "
              f"{path.stat().st_size / 1024:.0f} KB  {by_cut}")

    print("\nEsempio:")
    print((OUT / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
