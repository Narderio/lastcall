# LastCall – churn prevention for e-commerce

AMD Developer Hackathon ACT III · Track 3, Reinvent Commerce.

LastCall finds the customers of an online shop who are about to stop buying and writes a personal retention offer for each of them. The shop owner reviews every offer before anything goes out.

**Live demo:** <https://narderio.github.io/lastcall/> (the 200 offers are already written; live generation needs the local server, see [Run it](#run-it)).

## How it works

1. **Who is leaving.** A Random Forest scores every active customer from 0 to 100 using their purchase history (20 features, such as days since the last order compared with their usual rhythm).
2. **Who we write to.** Risk above 50 and at least two orders, or a single order above £500. The demo covers the first 200, ranked by risk × value.
3. **What we write.** One request per customer to **Qwen3.8-27B on AMD Token Factory (AMD GPUs)** returns the likely reason, one offer (similar products, discount, loyalty points or free shipping) and a short email. The answer is checked automatically (format, an email that matches the chosen offer, and that cites the customer's favourite product) and asked again if it fails.
4. **Review.** The dashboard (`demo/`) lets the shop owner approve, edit or reject each email. Nothing is sent automatically.

### What runs on AMD

The offer step, the LLM that writes the reason, the offer and the email, runs on AMD GPUs through [AMD Token Factory](https://notebooks.amd.com/tokenfactory) (OpenAI-compatible API). The dashboard's "Generate offer with AI" button calls the same model live.

## Data

[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI): about 1 million sales of a UK online wholesaler, Dec 2009 – Dec 2011. Customer IDs are anonymous numbers; the dataset has no names or emails.
Download the zip and extract `online_retail_II.xlsx` into `data/` (not included in the repo).

A customer has **left** if they make no purchase for 2 × their own average gap between purchase days, counted in whole days, at least 30 and at most 90 days (90 for customers with a single order). Reference date: 2011-09-10. The model is tested on customers it has never seen, with real outcomes only, no synthetic customers.

## Results

On 1,080 test customers never seen in training:

| Model | AUC |
|---|---|
| Random Forest, 20 features (chosen) | 0.760 |
| Best ensemble (soft voting) | 0.764 |
| Jev, no training | 0.720 |
| Rule "no purchase for 90 days" | 0.642 |

Risk bands for the 4,317 active customers (each scored by a model that never saw them): high risk (70–100) 1,177 customers, of which 77.5% really left; medium (40–69) 1,787, 57%; low (0–39) 1,353, 21%.

Offers for the demo's 200 customers: 200 of 200 answers in the right format, 23.5 minutes, about $0.03 in total.

## Run it

Python 3.12 with pandas, pyarrow, openpyxl, scikit-learn, xgboost, lightgbm, catboost, matplotlib. Run the scripts from the `churn/` folder.

```
python features.py        # one row per customer -> data/customers.csv
python rf_final.py        # risk scores -> data/customers_scored_rf.csv
python pipeline.py        # offers for the first 200 -> data/lastcall_offers.csv
python export_demo.py     # dashboard data -> demo/data.json
```

The Token Factory key goes in `churn/.env` (never committed):

```
TOKENFACTORY_API_KEY=...
```

Start the dashboard from the repository root and open <http://localhost:8766>:

```
python demo/server.py
```

The server keeps the key on your machine and generates offers live. The same pages are published as a static site on GitHub Pages (`.github/workflows/pages.yml`) with the 200 offers already written; only the live generation button needs the server.

## Scripts (`churn/`)

| Script | What it does |
|---|---|
| `features.py` | Cleans the transactions and builds one row per customer |
| `churn_rule_compare.py`, `churn_cap_compare.py` | Compare the fixed 90-day rule with the personal-threshold rule and different caps |
| `risk_model.py`, `risk_model_v2.py` | Logistic regression and gradient boosting baselines |
| `model_zoo.py`, `model_zoo_v2.py` | Compare 16 models, before and after 6 extra features |
| `rf_tuning.py` | Hyperparameter search for the Random Forest (no gain on the test set) |
| `rf_final.py` | Final Random Forest and risk scores for all active customers |
| `pipeline.py` | Picks the customers to write to and writes the offers with Qwen3.8-27B |
| `export_demo.py` | Exports the dashboard data (`demo/data.json`) |
| `jev_risk.py`, `jev_batch.py`, `jev_offers.py` | Early baseline with Jev (OpenRouter), not part of the final product |
| `llm.py` | Small OpenAI-compatible LLM client |

## Team

- Dario Nardella ([@Narderio](https://github.com/Narderio))
- Maria Labanca ([@marialabancaa](https://github.com/marialabancaa))

## License

MIT
