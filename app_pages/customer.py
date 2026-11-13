import altair as alt
import pandas as pd
import streamlit as st

from utils import SNAPSHOT_DATES, date_label, date_note, load_purchases, score_snapshot, show_reasons, show_verdict

st.title("Customer lookup")
st.markdown("**Choose a customer to see their churn risk, why, and their recent spending.**")

with st.container(horizontal=True, vertical_alignment="bottom"):
    date = st.selectbox("Date", SNAPSHOT_DATES[::-1], format_func=date_label, key="lookup_date", width=240)
    scored = score_snapshot(date)
    labels = {
        row["Customer ID"]: f"{row['Customer ID']} · {row['Country']} · {row['Churn probability']:.0%} risk"
        for _, row in scored.iterrows()
    }
    customer = st.selectbox(
        "Customer (highest risk first)", list(labels), format_func=labels.get, key="lookup_customer",
        help="Type a Customer ID to search.",
    )
st.caption(date_note(date))

row = scored.set_index("Customer ID").loc[customer]

left, right = st.columns(2)
with left:
    show_verdict(row["Churn probability"], row["Suggested action"] + ".")
    with st.container(border=True):
        show_reasons(row)
    if pd.notna(row.get("Churned")):
        happened = "stopped buying" if row["Churned"] == 1 else "ordered again"
        st.caption(f"What really happened: the customer {happened} in the next 90 days.")

with right.container(border=True):
    st.markdown("**Monthly spending over the past year**")
    purchases = load_purchases()
    history = purchases[
        (purchases["Customer ID"] == customer)
        & (purchases["InvoiceDate"] >= date - pd.Timedelta(days=365))
        & (purchases["InvoiceDate"] < date)
    ]
    spend = (history["Quantity"] * history["Price"]).groupby(history["InvoiceDate"].dt.to_period("M").dt.to_timestamp()).sum()
    months = pd.DataFrame({"Month": pd.date_range(date - pd.DateOffset(months=12), date - pd.DateOffset(months=1), freq="MS")})
    months["Spend"] = months["Month"].map(spend).fillna(0)
    st.altair_chart(
        alt.Chart(months).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#0F766E").encode(
            x=alt.X("yearmonth(Month):T", title=None, axis=alt.Axis(format="%b %y")),
            y=alt.Y("Spend:Q", title="Spend (£)"),
            tooltip=[alt.Tooltip("yearmonth(Month):T", title="Month"), alt.Tooltip("Spend:Q", format=",.0f")],
        ).properties(height=300)
    )
