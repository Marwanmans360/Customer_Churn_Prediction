"""Churn features calculated from purchase history.

The same steps as prepare_churn_data() in Churn_Feature_Preparation.ipynb, so
the app scores customers with exactly the feature definitions used in training.
"""
import pandas as pd

HISTORY_DAYS = 365
TARGET_DAYS = 90


def compute_features(purchases: pd.DataFrame, prediction_date) -> pd.DataFrame:
    """One row per customer who bought in the 365 days before prediction_date.

    The Churned column is added only when the full 90-day outcome is in the data.
    """
    prediction_date = pd.Timestamp(prediction_date).normalize()
    history_start = prediction_date - pd.Timedelta(days=HISTORY_DAYS)
    target_end = prediction_date + pd.Timedelta(days=TARGET_DAYS)

    if history_start < purchases["InvoiceDate"].min().normalize():
        raise ValueError("Not enough data for a full 365-day history.")

    history = purchases[
        (purchases["InvoiceDate"] >= history_start)
        & (purchases["InvoiceDate"] < prediction_date)
    ].copy()
    history["TransactionValue"] = history["Quantity"] * history["Price"]

    # Frequency and monetary value.
    customer_features = history.groupby("Customer ID").agg(
        InvoiceCountLast365Days=("Invoice", "nunique"),
        PurchaseAmountLast365Days=("TransactionValue", "sum"),
        DistinctProductCountLast365Days=("StockCode", "nunique"),
    ).reset_index()
    customer_features["AverageOrderValueLast365Days"] = (
        customer_features["PurchaseAmountLast365Days"] / customer_features["InvoiceCountLast365Days"]
    )

    # Recency and tenure.
    customer_dates = history.groupby("Customer ID").agg(
        FirstPurchaseDate=("InvoiceDate", "min"),
        LastPurchaseDate=("InvoiceDate", "max"),
    ).reset_index()
    customer_dates["DaysSinceLastPurchase"] = (
        prediction_date - customer_dates["LastPurchaseDate"].dt.normalize()
    ).dt.days
    customer_dates["DaysSinceFirstPurchase"] = (
        prediction_date - customer_dates["FirstPurchaseDate"].dt.normalize()
    ).dt.days

    # Purchase rhythm and OverdueRatio.
    history["PurchaseDay"] = history["InvoiceDate"].dt.normalize()
    purchase_days = (
        history.groupby("Customer ID")["PurchaseDay"].nunique().reset_index(name="PurchaseDayCount")
    )
    customer_rhythm = customer_dates.merge(purchase_days, on="Customer ID", how="left")
    span_days = (
        customer_rhythm["LastPurchaseDate"].dt.normalize()
        - customer_rhythm["FirstPurchaseDate"].dt.normalize()
    ).dt.days
    gap_count = (customer_rhythm["PurchaseDayCount"] - 1).replace(0, float("nan"))
    customer_rhythm["AverageDaysBetweenPurchases"] = span_days / gap_count
    customer_rhythm["OverdueRatio"] = (
        customer_rhythm["DaysSinceLastPurchase"] / customer_rhythm["AverageDaysBetweenPurchases"]
    )

    # Recent purchases.
    recent_90d = history[history["InvoiceDate"] >= prediction_date - pd.Timedelta(days=90)]
    recent_30d = history[history["InvoiceDate"] >= prediction_date - pd.Timedelta(days=30)]
    recent_90 = recent_90d.groupby("Customer ID").agg(
        InvoiceCountLast90Days=("Invoice", "nunique"),
        PurchaseAmountLast90Days=("TransactionValue", "sum"),
    ).reset_index()
    recent_30 = (
        recent_30d.groupby("Customer ID")["Invoice"].nunique().reset_index(name="InvoiceCountLast30Days")
    )
    customer_recent = customer_features[["Customer ID"]].merge(recent_90, on="Customer ID", how="left")
    customer_recent = customer_recent.merge(recent_30, on="Customer ID", how="left").fillna(0)

    # Country on the latest purchase.
    customer_country = (
        history.sort_values("InvoiceDate").groupby("Customer ID")["Country"].last().reset_index()
    )

    # Combine the features.
    features = customer_features.merge(
        customer_rhythm[[
            "Customer ID", "DaysSinceLastPurchase", "DaysSinceFirstPurchase",
            "AverageDaysBetweenPurchases", "OverdueRatio", "PurchaseDayCount",
        ]],
        on="Customer ID", how="left",
    )
    features = features.merge(customer_recent, on="Customer ID", how="left")
    features = features.merge(customer_country, on="Customer ID", how="left")
    features["PredictionDate"] = prediction_date

    # The outcome is known only when the full 90 days are in the data.
    if target_end <= purchases["InvoiceDate"].max():
        future_customers = purchases[
            (purchases["InvoiceDate"] >= prediction_date) & (purchases["InvoiceDate"] < target_end)
        ]["Customer ID"].unique()
        features["Churned"] = (~features["Customer ID"].isin(future_customers)).astype(int)

    return features
