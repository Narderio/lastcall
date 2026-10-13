"""Feature per cliente ed etichetta churn dal dataset Online Retail II.

Uso: python features.py   (dalla cartella churn/, dati in ../data/)
"""
import importlib.util
import time
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
XLSX = DATA / "online_retail_II.xlsx"
HORIZON = pd.Timedelta(days=90)
YEAR = pd.Timedelta(days=365)
HAS_PARQUET = importlib.util.find_spec("pyarrow") is not None
CACHE = DATA / ("transactions.parquet" if HAS_PARQUET else "transactions.csv")


def load_transactions() -> pd.DataFrame:
    if CACHE.exists():
        if HAS_PARQUET:
            return pd.read_parquet(CACHE)
        return pd.read_csv(CACHE, parse_dates=["InvoiceDate"], dtype={"Invoice": str, "StockCode": str})
    sheets = pd.read_excel(XLSX, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    first, second = sheets.values()
    # I due fogli si sovrappongono (1-9 dic 2010): dal primo teniamo solo le date prima dell'inizio del secondo
    overlap = first["InvoiceDate"] >= second["InvoiceDate"].min()
    print(f"Righe sovrapposte tolte dal primo foglio: {overlap.sum()}")
    df = pd.concat([first[~overlap], second], ignore_index=True)
    df["Description"] = df["Description"].astype(str)
    if HAS_PARQUET:
        df.to_parquet(CACHE, index=False)
    else:
        df.to_csv(CACHE, index=False)
    print(f"Copia salvata in {CACHE}")
    return df


def clean(df: pd.DataFrame):
    df = df.dropna(subset=["Customer ID"]).copy()
    df["customer_id"] = df["Customer ID"].astype(int).astype(str)
    cancelled = df["Invoice"].astype(str).str.startswith("C")
    returns = df[cancelled]
    buys = df[~cancelled & (df["Quantity"] > 0) & (df["Price"] > 0)].copy()
    buys["amount"] = buys["Quantity"] * buys["Price"]
    return buys, returns


def build_features(buys: pd.DataFrame, returns: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    past = buys[buys["InvoiceDate"] < cutoff]
    last12 = past[past["InvoiceDate"] >= cutoff - YEAR]
    prev12 = past[(past["InvoiceDate"] >= cutoff - 2 * YEAR) & (past["InvoiceDate"] < cutoff - YEAR)]
    ret12 = returns[(returns["InvoiceDate"] < cutoff) & (returns["InvoiceDate"] >= cutoff - YEAR)]

    g = past.groupby("customer_id")
    f = pd.DataFrame({
        "days_since_last_purchase": (cutoff - g["InvoiceDate"].max()).dt.days,
        "first_purchase_days_ago": (cutoff - g["InvoiceDate"].min()).dt.days,
        "n_orders": g["Invoice"].nunique(),
        "total_spend": g["amount"].sum(),
        "country": g["Country"].agg(lambda s: s.mode().iat[0]),
    })
    f["avg_order_value"] = f["total_spend"] / f["n_orders"]

    for name, part in (("last_12m", last12), ("prev_12m", prev12)):
        pg = part.groupby("customer_id")
        n = pg["Invoice"].nunique()
        f[f"n_orders_{name}"] = n.reindex(f.index, fill_value=0)
        f[f"avg_order_value_{name}"] = (pg["amount"].sum() / n).reindex(f.index)

    f["n_returns_last_12m"] = ret12.groupby("customer_id")["Invoice"].nunique().reindex(f.index, fill_value=0)

    qty = past.groupby(["customer_id", "Description"])["Quantity"].sum().reset_index()
    fav = qty.sort_values("Quantity", ascending=False).drop_duplicates("customer_id").set_index("customer_id")
    f["favourite_category"] = fav["Description"]

    future = buys[(buys["InvoiceDate"] >= cutoff) & (buys["InvoiceDate"] < cutoff + HORIZON)]
    f["churned"] = (~f.index.isin(future["customer_id"].unique())).astype(int)

    cols = ["days_since_last_purchase", "n_orders", "n_orders_last_12m", "n_orders_prev_12m",
            "total_spend", "avg_order_value", "avg_order_value_last_12m", "avg_order_value_prev_12m",
            "n_returns_last_12m", "favourite_category", "country", "first_purchase_days_ago", "churned"]
    return f[cols].round(2).reset_index()


if __name__ == "__main__":
    t0 = time.perf_counter()
    df = load_transactions()
    t1 = time.perf_counter()
    print(f"Transazioni caricate: {len(df)} righe in {t1 - t0:.1f}s")

    buys, returns = clean(df)
    last = df["InvoiceDate"].max()
    cutoff = last.normalize() - HORIZON
    print(f"Prima data: {df['InvoiceDate'].min()}  ultima data: {last}  CUTOFF: {cutoff.date()}")
    print(f"Acquisti puliti: {len(buys)}  righe di resi: {len(returns)}")

    customers = build_features(buys, returns, cutoff)
    customers.to_csv(DATA / "customers.csv", index=False)
    t2 = time.perf_counter()

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"\nClienti: {len(customers)}")
    print(f"Churned: {customers['churned'].mean():.1%}")
    print("\nStatistiche:")
    print(customers.describe().round(2).T.to_string())
    print("\nEsempi:")
    print(customers.head(3).to_string())
    print(f"\nTempo feature: {t2 - t1:.1f}s  totale: {t2 - t0:.1f}s  salvato in {DATA / 'customers.csv'}")
