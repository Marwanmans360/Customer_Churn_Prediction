"""Shared data loading, scoring and display helpers for the customer churn app."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import compute_features

ROOT = Path(__file__).parent

LATEST_DATE = pd.Timestamp("2011-12-01")
SNAPSHOT_DATES = list(pd.date_range("2010-12-01", LATEST_DATE, freq="MS"))
TRAINING_MONTHS_END = pd.Timestamp("2011-09-01")  # The saved model was refit on Dec 2010 - Sep 2011.

# Reported once on the reserved August-September 2011 test months by
# Churn_Preprocessing_and_Machine_Learning.ipynb (10-feature Random Forest).
TEST = {
    "customers": 8_648, "leavers": 4_642, "caught": 4_342, "flagged": 6_670,
    "recall": 0.9354, "precision": 0.6510, "accuracy": 0.6961, "baseline_accuracy": 0.5368,
}

RISK_TIERS = ["High", "Medium", "Low"]
TIER_COLORS = {"High": "red", "Medium": "orange", "Low": "green"}


# ---------- Data and model ----------
@st.cache_data
def load_purchases() -> pd.DataFrame:
    """The cleaned purchases saved by 1_Data_Cleaning.ipynb."""
    return pd.read_csv(
        ROOT / "data" / "cleaned_purchases.csv.gz",
        dtype={"Invoice": "string", "StockCode": "string", "Customer ID": "string"},
        parse_dates=["InvoiceDate"],
    )


@st.cache_data
def load_training_data() -> pd.DataFrame:
    """The monthly customer table saved by 2_Feature_Engineering_and_EDA.ipynb."""
    return pd.read_csv(ROOT / "data" / "churn_data.csv", dtype={"Customer ID": "string"}, parse_dates=["PredictionDate"])


@st.cache_resource
def load_model():
    """The pipeline saved by 3_Machine_Learning.ipynb."""
    return joblib.load(ROOT / "models" / "churn_model.pkl")


def churn_probability(model, rows: pd.DataFrame) -> np.ndarray:
    """Probability of Churned = 1, using exactly the model's input columns."""
    return model.predict_proba(rows[list(model.feature_names_in_)])[:, 1]


def risk_tier(probability: float) -> str:
    if probability >= 0.75:
        return "High"
    if probability >= 0.5:
        return "Medium"
    return "Low"


def suggested_action(tier: str, spend: float, typical_spend: float) -> str:
    if tier == "High" and spend >= typical_spend:
        return "Call now: valuable customer at risk"
    if tier == "High":
        return "Send a win-back offer"
    if tier == "Medium":
        return "Check in or send a reminder"
    return "No action needed"


@st.cache_data(max_entries=16)
def score_snapshot(prediction_date: pd.Timestamp) -> pd.DataFrame:
    """Calculate every active customer's features on a date from the invoices and score them."""
    scored = compute_features(load_purchases(), prediction_date)
    scored["Churn probability"] = churn_probability(load_model(), scored)
    scored["Risk"] = scored["Churn probability"].map(risk_tier)
    typical_spend = scored["PurchaseAmountLast365Days"].median()
    scored["Suggested action"] = [
        suggested_action(tier, spend, typical_spend)
        for tier, spend in zip(scored["Risk"], scored["PurchaseAmountLast365Days"])
    ]
    return scored.sort_values("Churn probability", ascending=False).reset_index(drop=True)


def date_label(date) -> str:
    date = pd.Timestamp(date)
    label = f"{date.day} {date:%B %Y}"
    return f"{label} (today)" if date == LATEST_DATE else label


def date_note(date) -> str:
    if pd.Timestamp(date) > TRAINING_MONTHS_END:
        return "A real forecast: the next 90 days are not in the data yet."
    return "A past date the model was trained on, shown as a demonstration."


# ---------- Explaining a prediction in plain words ----------
@st.cache_data
def typical_stayer() -> pd.Series:
    """Median values of customers who ordered again within 90 days."""
    data = load_training_data()
    return data[data["Churned"] == 0].median(numeric_only=True)


def reasons(row: pd.Series) -> list[str]:
    """Short, coloured statements comparing a customer with a typical customer who stayed."""
    typical = typical_stayer()

    def line(bad: bool, text: str) -> str:
        icon = ":material/arrow_upward:" if bad else ":material/arrow_downward:"
        return f":{'red' if bad else 'green'}[{icon}] {text}"

    days, usual_days = row["DaysSinceLastPurchase"], typical["DaysSinceLastPurchase"]
    orders, usual_orders = row["InvoiceCountLast365Days"], typical["InvoiceCountLast365Days"]
    spend, usual_spend = row["PurchaseAmountLast365Days"], typical["PurchaseAmountLast365Days"]
    items = [
        line(days > usual_days, f"Last order **{days:.0f} days ago** (customers who stay: about {usual_days:.0f})"),
        line(orders < usual_orders, f"**{orders:.0f} orders** in the past year (customers who stay: about {usual_orders:.0f})"),
        line(spend < usual_spend, f"**£{spend:,.0f}** spent in the past year (customers who stay: about £{usual_spend:,.0f})"),
    ]
    if pd.isna(row["OverdueRatio"]):
        items.append(line(True, "Only **one purchase day** so far, so no buying habit yet"))
    else:
        ratio = row["OverdueRatio"]
        if ratio > 1:
            items.append(line(True, f"**{ratio:.1f}×** their usual gap has passed since the last order"))
        else:
            items.append(line(False, f"Not due yet: **{ratio:.1f}×** their usual gap since the last order"))
    return items


def show_reasons(row: pd.Series) -> None:
    st.markdown("**Why**  \n" + "  \n".join(reasons(row)))
    st.caption(":red[:material/arrow_upward:] raises the risk · :green[:material/arrow_downward:] lowers it")


def show_verdict(probability: float, action: str | None = None) -> None:
    tier = risk_tier(probability)
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.metric("Churn risk", f"{probability:.0%}")
            st.badge(f"{tier} risk", color=TIER_COLORS[tier], icon=":material/flag:")
        st.progress(float(probability))
        if probability >= 0.5:
            st.markdown(f":red[**Likely to stop buying**] in the next 90 days. {action or ''}")
        else:
            st.markdown(f":green[**Likely to order again**] in the next 90 days. {action or ''}")


# ---------- Simulator examples (real customers on 1 September 2011) ----------
EXAMPLE_DATE = pd.Timestamp("2011-09-01")
EXAMPLES = {
    "Loyal wholesaler": {
        "customer": "15005",
        "summary": "Orders every two weeks, last bought 9 days ago.",
        "try": "Set days since last purchase to 150, with 0 orders and £0 in the last 90 days.",
        "discuss": "Frequency and spending protect a customer, but even a loyal customer becomes a risk once they go quiet.",
    },
    "Regular going quiet": {
        "customer": "17849",
        "summary": "Usually orders every 38 days, but nothing for 119 days.",
        "try": "Set days since last purchase to 20, with 1 order of £150 in the last 90 days.",
        "discuss": "The case the tool is built for: a regular customer breaking their habit. A call now could win them back.",
    },
    "One-time buyer": {
        "customer": "12373",
        "summary": "A single order from Austria about seven months ago.",
        "try": "Change the country to United Kingdom or Germany.",
        "discuss": "One-order customers leave most often. Country barely changes the risk; behaviour matters far more.",
    },
    "Brand-new customer": {
        "customer": "15701",
        "summary": "First and only order 11 days ago.",
        "try": "Make it 2 orders on 2 different days, first purchase 40 days ago.",
        "discuss": "New customers have no habit yet, so the model is unsure. This one ordered again: a cheap false alarm.",
    },
    "Surprise churner": {
        "customer": "13082",
        "summary": "12 orders last year and a recent order. Looks healthy.",
        "try": "Nothing in the history explains it; this customer still left.",
        "discuss": "The model only sees purchases. It cannot know about closures, competitors or complaints.",
    },
}


@st.cache_data
def example_rows() -> pd.DataFrame:
    scored = score_snapshot(EXAMPLE_DATE)
    ids = [example["customer"] for example in EXAMPLES.values()]
    return scored[scored["Customer ID"].isin(ids)].set_index("Customer ID")


def svg(name: str) -> str:
    return (ROOT / "assets" / name).read_text(encoding="utf-8")
