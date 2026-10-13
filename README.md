# LastCall – churn prevention for e-commerce

AMD Developer Hackathon ACT III · Reinvent Commerce track.

Find the customers of an online shop who are about to stop buying, explain why, and suggest a retention offer.

- **Risk:** a classic model (Random Forest) trained on each customer's purchase history.
- **Reason and offer:** a decision model (Jev via OpenRouter today, the open-weight Kev on an AMD MI300X GPU next).

Work in progress.

## Data

[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI): about 1 million sales of a UK online shop, Dec 2009 – Dec 2011.
Download the zip and extract `online_retail_II.xlsx` into `data/` (not included in the repo).

A customer is labelled as **churned** if they make no purchase within a personal threshold after the reference date (2011-09-10):
2 × their average gap between purchase days, at least 30 and at most 90 days.

## Scripts (`churn/`)

| Script | What it does |
|---|---|
| `features.py` | Cleans the transactions and builds one row per customer (`data/customers.csv`) |
| `churn_rule_compare.py` | Compares the fixed 90-day churn rule with the personal-threshold rule |
| `risk_model.py`, `risk_model_v2.py` | Logistic regression and gradient boosting baselines |
| `model_zoo.py`, `model_zoo_v2.py` | Compares 16 models (statistical and ensembles), before and after 6 extra features |
| `jev_risk.py`, `jev_batch.py`, `jev_offers.py` | Churn probability, reason and retention offer from Jev (OpenRouter Decisions API) |
| `build_kev_data.py` | Training data for fine-tuning Kev |
| `llm.py` | One LLM client for OpenRouter (development) and vLLM on AMD (final) |

## Setup

Python 3.12 with pandas, pyarrow, openpyxl, scikit-learn, xgboost, lightgbm, catboost, matplotlib.

The OpenRouter key goes in `churn/.env` (never committed):

```
OPENROUTER_API_KEY=...
```

## Results so far (1,080 test customers)

| Model | AUC |
|---|---|
| Random Forest, 20 features | 0.760 |
| Best ensemble (soft voting) | 0.764 |
| Jev, no training | 0.720 |
| Rule "no purchase for 90 days" | 0.642 |

## Team

- Dario Nardella ([@Narderio](https://github.com/Narderio))
- Maria Labanca ([@marialabancaa](https://github.com/marialabancaa))

## License

MIT
