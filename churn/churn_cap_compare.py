"""Regola nuova (2 x intervallo medio, minimo 30, giorni interi) con tetti diversi: 90, 120, 150, 180.

Solo clienti attivi. Confronto con la regola vecchia (90 giorni fissi). Una combinazione è "non calcolabile"
se taglio + tetto va oltre l'ultima data dei dati. Scrive ../data/churn_cap_compare.png.
Uso: python churn_cap_compare.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import DATA, clean, load_transactions

CUTOFFS = ("2011-03-14", "2011-06-12", "2011-09-10")
CAPS = (90, 120, 150, 180)
OLD = 90


def per_customer(buys, cutoff):
    past = buys[buys["InvoiceDate"] < cutoff]
    active = past.groupby("customer_id")["InvoiceDate"].max() >= cutoff - pd.Timedelta(days=365)
    days = past.assign(day=past["InvoiceDate"].dt.normalize()).groupby("customer_id")["day"]
    gap = days.apply(lambda s: np.diff(np.sort(s.unique())).astype("timedelta64[D]").astype(float).mean()
                     if s.nunique() > 1 else np.nan)
    after = buys[buys["InvoiceDate"] >= cutoff].groupby("customer_id")["InvoiceDate"].min().dt.normalize()
    out = pd.DataFrame({"gap": gap, "next": (after - cutoff).dt.days}).loc[active[active].index]
    return out


if __name__ == "__main__":
    buys, _ = clean(load_transactions())
    last_day = buys["InvoiceDate"].max().normalize()
    rows = []
    for c in CUTOFFS:
        cut = pd.Timestamp(c)
        d = per_customer(buys, cut)
        old = ~(d["next"] < OLD)
        for cap in CAPS:
            if cut + pd.Timedelta(days=cap) > last_day:
                rows.append({"taglio": c, "tetto": cap, "calcolabile": False})
                continue
            soglia = (2 * d["gap"]).clip(30, cap).fillna(cap)
            new = ~(d["next"] < soglia)
            rows.append({"taglio": c, "tetto": cap, "calcolabile": True, "n": len(d),
                         "pct_vecchia": old.mean() * 100, "pct_nuova": new.mean() * 100,
                         "soglia_sopra_90": int((soglia > 90).sum()),
                         "da_0_a_1": int((~old & new).sum()), "da_1_a_0": int((old & ~new).sum())})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(1).to_string(index=False))

    # Grafico: % andati via per data, una barra per variante; tratteggio = non calcolabile
    colors = {"vecchia 90": "#8a8985", 90: "#2a78d6", 120: "#eb6834", 150: "#1baf7a", 180: "#eda100"}
    labels = ["vecchia 90"] + list(CAPS)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    w = 0.16
    for i, c in enumerate(CUTOFFS):
        sub = res[res["taglio"] == c].set_index("tetto")
        vals = [sub.loc[90, "pct_vecchia"]] + [sub.loc[k, "pct_nuova"] if sub.loc[k, "calcolabile"] else None for k in CAPS]
        for j, (lab, v) in enumerate(zip(labels, vals)):
            x = i + (j - 2) * w
            name = lab if lab == "vecchia 90" else f"nuova, tetto {lab}"
            if v is None:
                ax.text(x, 2, "n.c.", ha="center", va="bottom", fontsize=8, color="#52514e", rotation=90)
                continue
            ax.bar(x, v, w - 0.02, color=colors[lab], label=name if i == 0 else None)
            ax.text(x, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="#0b0b0b")
    ax.set_xticks(range(len(CUTOFFS)), [pd.Timestamp(c).strftime("%d %b %Y") for c in CUTOFFS])
    ax.set_ylabel("% clienti attivi andati via")
    ax.set_ylim(0, 100)
    ax.set_title("Quanti clienti risultano andati via, per regola e tetto", loc="left", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper left", bbox_to_anchor=(0, -0.08))
    fig.text(0.01, 0.01, "n.c. = non calcolabile: i dati finiscono il 9 dic 2011", fontsize=8, color="#52514e")
    fig.tight_layout()
    fig.savefig(DATA / "churn_cap_compare.png")
    print(f"\nGrafico: {DATA / 'churn_cap_compare.png'}")
