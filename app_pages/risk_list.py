import streamlit as st

from utils import RISK_TIERS, SNAPSHOT_DATES, date_label, date_note, score_snapshot

st.title("At-risk customers")
st.markdown("**Pick a date, review who is likely to stop buying, and download the list for the sales team.**")

with st.container(horizontal=True, vertical_alignment="bottom"):
    date = st.selectbox("Date", SNAPSHOT_DATES[::-1], format_func=date_label, key="risk_date", width=240)
    tiers = st.pills("Show risk", RISK_TIERS, selection_mode="multi", default=["High", "Medium"], key="risk_tiers")
st.caption(date_note(date))

scored = score_snapshot(date)
at_risk = scored[scored["Churn probability"] >= 0.5]

with st.container(horizontal=True):
    st.metric("Active customers", f"{len(scored):,}", border=True)
    st.metric("Likely to stop buying", f"{len(at_risk):,}", border=True)
    st.metric(
        "Their spending last year", f"£{at_risk['PurchaseAmountLast365Days'].sum():,.0f}",
        help="What the at-risk customers spent in the previous 365 days: the business at stake.", border=True,
    )

view = scored[scored["Risk"].isin(tiers)] if tiers else scored
columns = [
    "Customer ID", "Churn probability", "Risk", "Suggested action",
    "DaysSinceLastPurchase", "InvoiceCountLast365Days", "PurchaseAmountLast365Days", "Country",
]
st.dataframe(
    view[columns],
    hide_index=True,
    column_config={
        "Churn probability": st.column_config.ProgressColumn("Churn risk", format="percent", min_value=0.0, max_value=1.0),
        "DaysSinceLastPurchase": st.column_config.NumberColumn("Days since last order"),
        "InvoiceCountLast365Days": st.column_config.NumberColumn("Orders (past year)"),
        "PurchaseAmountLast365Days": st.column_config.NumberColumn("Spent (past year)", format="£%.0f"),
    },
)

with st.container(horizontal=True):
    st.download_button(
        f"Download list ({len(view):,} customers)",
        view[columns].to_csv(index=False).encode("utf-8"),
        file_name=f"at_risk_customers_{date:%Y_%m_%d}.csv",
        mime="text/csv",
        icon=":material/download:",
        type="primary",
    )
    st.page_link("app_pages/customer.py", label="Check one customer", icon=":material/person_search:")

st.caption("Risk levels: High = 75% or more, Medium = 50-75%, Low = below 50%.")
