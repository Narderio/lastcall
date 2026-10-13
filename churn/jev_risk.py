#!/usr/bin/env python3
"""Rischio churn di un cliente con Jev (TypeSafe) via OpenRouter Decisions API.

Uso (sul tuo computer, la chiave non va mai nel codice né nel repo):
  export OPENROUTER_API_KEY=sk-or-...
  python3 jev_risk.py

Formato API: https://openrouter.ai/docs/guides/community/jev-tutorial
Più avanti la stessa funzione punterà a Kev su GPU AMD.
"""
import json
import os
import urllib.request

URL = os.environ.get("DECISIONS_URL", "https://openrouter.ai/api/alpha/decisions")
MODEL = os.environ.get("DECISIONS_MODEL", "typesafe/jev-1.13")

QUESTIONS = {
    "will_churn": {
        "type": "noul",
        "instructions": "Will this customer stop buying from the shop in the next 90 days?",
        "criteria": {
            "true": "The customer is likely to make no further purchase in the next 90 days.",
            "false": "The customer is likely to buy again within 90 days.",
        },
    },
    "risk_level": {
        "type": "score",
        "instructions": "How high is the risk that this customer is leaving?",
        "criteria": ["Loyal, buying regularly", "Slowing down", "Clearly at risk", "Almost gone"],
    },
    "main_reason": {
        "type": "choice",
        "instructions": "What is the most likely reason for the change in buying behaviour?",
        "criteria": {
            "price": "Spending per order is dropping, they may find cheaper alternatives.",
            "lost_interest": "Orders are less frequent and the gap since last purchase is growing.",
            "seasonal": "The customer only buys in specific periods of the year.",
            "bad_experience": "Returns or cancellations suggest dissatisfaction.",
            "none": "No real sign of leaving.",
        },
    },
}

# Domanda separata: senza il nome del prodotto preferito nello state, che la spingeva sempre verso
# "favourite_category", e con la probabilità di churn già stimata per poter scegliere no_offer.
OFFER_QUESTIONS = {
    "best_offer": {
        "type": "choice",
        "instructions": "Which retention offer fits this customer best? Base the choice on the observable numbers, "
                        "not on the product names.",
        "criteria": {
            "discount": "Average order value in the last 12 months is clearly lower than in the previous 12 months "
                        "(spending per order dropped) and the churn probability is not low.",
            "free_shipping": "Small average order value but frequent orders: shipping cost weighs on each order.",
            "loyalty_points": "Long-standing customer (first purchase long ago) with many orders overall, "
                              "worth rewarding to keep them engaged.",
            "similar_products": "Orders are becoming less frequent (fewer orders in the last 12 months than before, "
                                "or many days since the last purchase): suggest products similar to past purchases.",
            "no_offer": "Churn probability is low and buying behaviour is stable or growing: no offer needed.",
        },
    },
}


def decide(state: dict, questions: dict = QUESTIONS) -> dict:
    req = urllib.request.Request(
        URL,
        data=json.dumps({"model": MODEL, "state": state, "questions": questions}).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def offer_state(state: dict, churn_probability: float) -> dict:
    return {k: v for k, v in state.items() if k != "favourite_category"} | {
        "churn_probability": round(churn_probability, 2)}


def recommend(state: dict) -> dict:
    """Rischio e motivo, poi offerta: due chiamate, risposte unite."""
    risk = decide(state)
    offer = decide(offer_state(state, risk["answers"]["will_churn"]["noul"]), OFFER_QUESTIONS)
    cost = (risk.get("usage", {}).get("cost") or 0) + (offer.get("usage", {}).get("cost") or 0)
    return {"model": risk.get("model"), "answers": risk["answers"] | offer["answers"], "usage": {"cost": cost}}


if __name__ == "__main__":
    # Cliente di esempio inventato, solo per provare la chiamata
    cliente = {
        "customer_id": "esempio-001",
        "days_since_last_purchase": 74,
        "orders_last_12_months": 9,
        "orders_previous_12_months": 15,
        "avg_order_value_eur": 42.5,
        "avg_order_value_previous_year_eur": 58.0,
        "favourite_category": "coffee and tea",
        "returns_last_12_months": 2,
    }
    out = recommend(cliente)
    a = out["answers"]
    print(f"Modello: {out.get('model')}")
    print(f"Probabilità di abbandono: {a['will_churn']['noul']:.0%}")
    print(f"Livello di rischio: {a['risk_level']['score']:.2f} su 3")
    print(f"Motivo: {a['main_reason']['choice']}  {a['main_reason']['probabilities']}")
    print(f"Offerta: {a['best_offer']['choice']}  {a['best_offer']['probabilities']}")
    print(f"Costo: {out.get('usage', {}).get('cost')}")
