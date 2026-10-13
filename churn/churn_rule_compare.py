"""Confronto tra due definizioni di "cliente andato via" su tre date di taglio.

VECCHIA: nessun acquisto nei 90 giorni dopo il taglio.
NUOVA:   nessun acquisto entro una soglia personale = 2 x intervallo medio tra ordini,
         limitata tra 30 e 90 giorni (90 per chi ha un solo giorno di acquisto prima del taglio).
Uso: python churn_rule_compare.py   (dopo features.py; scrive ../data/churn_rule_compare.csv)
"""
import numpy as np
import pandas as pd

from features import DATA, clean, load_transactions

CUTOFFS = ("2011-03-14", "2011-06-12", "2011-09-10")
HORIZON = 90


def compare(buys: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    past = buys[buys["InvoiceDate"] < cutoff]
    g = past.groupby("customer_id")
    days = past.assign(day=past["InvoiceDate"].dt.normalize()).groupby("customer_id")["day"]
    gaps = days.apply(lambda s: np.diff(np.sort(s.unique())).astype("timedelta64[D]").astype(float).mean()
                      if s.nunique() > 1 else np.nan)

    future = buys[(buys["InvoiceDate"] >= cutoff) & (buys["InvoiceDate"] < cutoff + pd.Timedelta(days=HORIZON))]
    # Giorni interi: conta la data dell'acquisto, non l'ora
    first_after = (future.groupby("customer_id")["InvoiceDate"].min().dt.normalize() - cutoff).dt.days

    out = pd.DataFrame({
        "n_ordini": g["Invoice"].nunique(),
        "attivo": (g["InvoiceDate"].max() >= cutoff - pd.Timedelta(days=365)).astype(int),
        "intervallo_medio": gaps.round(1),
    })
    out["soglia"] = (2 * out["intervallo_medio"]).clip(30, HORIZON).fillna(HORIZON).round(1)
    out["giorni_al_prossimo"] = first_after.reindex(out.index).round(1)
    out["vecchio"] = out["giorni_al_prossimo"].isna().astype(int)
    out["nuovo"] = ~(out["giorni_al_prossimo"] < out["soglia"])
    out["nuovo"] = out["nuovo"].astype(int)
    out.insert(0, "data_taglio", cutoff.date())
    return out.reset_index()


def summary(d: pd.DataFrame) -> str:
    return (f"n {len(d)}, vecchia {d['vecchio'].mean():.1%}, nuova {d['nuovo'].mean():.1%}, "
            f"0->1 {((d['vecchio'] == 0) & (d['nuovo'] == 1)).sum()}, 1->0 {((d['vecchio'] == 1) & (d['nuovo'] == 0)).sum()}")


if __name__ == "__main__":
    buys, _ = clean(load_transactions())
    res = pd.concat([compare(buys, pd.Timestamp(c)) for c in CUTOFFS], ignore_index=True)
    cols = ["customer_id", "data_taglio", "attivo", "n_ordini", "intervallo_medio", "soglia", "vecchio", "nuovo"]
    res[cols].to_csv(DATA / "churn_rule_compare.csv", index=False)

    print("1) Per data di taglio")
    for c, d in res.groupby("data_taglio"):
        print(f"{c}  tutti:  {summary(d)}")
        print(f"{c}  attivi: {summary(d[d['attivo'] == 1])}")

    sep = res[(res["data_taglio"] == pd.Timestamp(CUTOFFS[-1]).date()) & (res["attivo"] == 1)]
    s = sep["soglia"]
    print("\n2) Soglia a settembre, clienti attivi")
    print(f"30: {(s == 30).sum()}, 31-45: {((s > 30) & (s <= 45)).sum()}, 46-60: {((s > 45) & (s <= 60)).sum()}, "
          f"61-89: {((s > 60) & (s < 90)).sum()}, 90: {(s == 90).sum()} "
          f"(di cui un solo giorno di acquisto: {sep['intervallo_medio'].isna().sum()})")

    print("\n3) Esempi di settembre in cui le regole non sono d'accordo")
    for v, n in ((0, 1), (1, 0)):
        ex = sep[(sep["vecchio"] == v) & (sep["nuovo"] == n)].head(2)
        for _, r in ex.iterrows():
            nxt = "nessuno entro 90" if pd.isna(r["giorni_al_prossimo"]) else f"{r['giorni_al_prossimo']:.0f}"
            print(f"{r['customer_id']}: ordini {r['n_ordini']}, intervallo medio {r['intervallo_medio']}, soglia {r['soglia']}, "
                  f"prossimo acquisto dopo {nxt} giorni, vecchio {r['vecchio']}, nuovo {r['nuovo']}")
