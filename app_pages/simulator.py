import numpy as np
import pandas as pd
import streamlit as st

from utils import EXAMPLES, churn_probability, example_rows, load_model, load_purchases, show_reasons, show_verdict

# Widget key -> (feature column, type)
INPUTS = {
    "sim_invoices_365": ("InvoiceCountLast365Days", int),
    "sim_amount_365": ("PurchaseAmountLast365Days", float),
    "sim_products_365": ("DistinctProductCountLast365Days", int),
    "sim_days_last": ("DaysSinceLastPurchase", int),
    "sim_days_first": ("DaysSinceFirstPurchase", int),
    "sim_purchase_days": ("PurchaseDayCount", int),
    "sim_invoices_90": ("InvoiceCountLast90Days", int),
    "sim_amount_90": ("PurchaseAmountLast90Days", float),
    "sim_country": ("Country", str),
}


def example_values(name: str) -> dict:
    row = example_rows().loc[EXAMPLES[name]["customer"]]
    return {key: kind(round(row[column], 2)) if kind is float else kind(row[column]) for key, (column, kind) in INPUTS.items()}


def load_example() -> None:
    name = st.session_state.get("sim_example")
    if name:
        st.session_state.update(example_values(name))


if "sim_example" not in st.session_state:
    st.session_state.sim_example = next(iter(EXAMPLES))
    load_example()

st.title("What-if")
st.markdown("**Start from a real customer, change their history, and watch the churn risk update.**")

with st.container(horizontal=True, vertical_alignment="bottom"):
    name = st.pills("Example customer", list(EXAMPLES), key="sim_example", on_change=load_example)
    st.button("Reset example", icon=":material/restart_alt:", on_click=load_example, type="tertiary")
if name:
    example = EXAMPLES[name]
    st.markdown(f"{example['summary']}  \n:material/lightbulb: **Try this:** {example['try']}")
    with st.expander("Talking point"):
        st.write(example["discuss"])

state = st.session_state

# Keep linked inputs consistent automatically, before the widgets are drawn.
no_recent_orders = state.sim_days_last > 90
if no_recent_orders:
    state.sim_invoices_90 = 0
    state.sim_amount_90 = 0.0
one_order_day = state.sim_purchase_days == 1
if one_order_day:
    state.sim_days_first = state.sim_days_last

countries = load_purchases()["Country"].value_counts().index.tolist()
locked_90 = "Locked at 0: the last order was more than 90 days ago. Set 'Days since last order' to 90 or less to add recent orders."
left, right = st.columns([3, 2])

with left.container(border=True):
    c1, c2, c3 = st.columns(3)
    c1.number_input("Orders (past year)", min_value=1, max_value=400, step=1, key="sim_invoices_365")
    c2.number_input("Spent, past year (£)", min_value=0.0, max_value=500_000.0, step=100.0, format="%.2f", key="sim_amount_365")
    c3.number_input("Different products", min_value=1, max_value=5000, step=5, key="sim_products_365")

    c1, c2, c3 = st.columns(3)
    c1.number_input("Days since last order", min_value=1, max_value=365, step=1, key="sim_days_last")
    c2.number_input(
        "Days since first order", min_value=1, max_value=365, step=1, key="sim_days_first", disabled=one_order_day,
        help="Same as the last order while there is only one day with an order." if one_order_day else "Within the past 365 days.",
    )
    c3.number_input("Days with an order", min_value=1, max_value=365, step=1, key="sim_purchase_days",
                    help="Several orders on the same day count as one day.")

    c1, c2, c3 = st.columns(3)
    c1.number_input("Orders (last 90 days)", min_value=0, max_value=400, step=1, key="sim_invoices_90",
                    disabled=no_recent_orders, help=locked_90 if no_recent_orders else None)
    c2.number_input("Spent, last 90 days (£)", min_value=0.0, max_value=500_000.0, step=50.0, format="%.2f",
                    key="sim_amount_90", disabled=no_recent_orders, help=locked_90 if no_recent_orders else None)
    c3.selectbox("Country", countries, key="sim_country")

# Remaining combinations that cannot happen, each with the fix.
problems = []
if state.sim_invoices_90 > state.sim_invoices_365:
    problems.append(f"Orders in the last 90 days are part of the past year: set **Orders (past year)** to at least {state.sim_invoices_90}.")
if state.sim_amount_90 > state.sim_amount_365:
    problems.append(f"Spending in the last 90 days is part of the past year: set **Spent, past year** to at least £{state.sim_amount_90:,.2f}.")
if state.sim_purchase_days > state.sim_invoices_365:
    problems.append(f"Each day with an order needs at least one order: set **Orders (past year)** to at least {state.sim_purchase_days}.")
if state.sim_purchase_days > 1 and state.sim_purchase_days > state.sim_days_first - state.sim_days_last + 1:
    needed = state.sim_days_last + state.sim_purchase_days - 1
    problems.append(f"{state.sim_purchase_days} days with an order need more time between the first and last order: set **Days since first order** to at least {needed}.")
if state.sim_days_last <= 90 and state.sim_invoices_90 == 0:
    problems.append(f"The last order was {state.sim_days_last} days ago, inside the last 90 days: set **Orders (last 90 days)** to at least 1.")

with right:
    if problems:
        st.warning("**These numbers don't fit together yet.**\n\n" + "\n".join(f"- {p}" for p in problems), icon=":material/edit:")
        st.stop()

    # Same formulas as the feature notebook.
    if state.sim_purchase_days > 1:
        average_gap = (state.sim_days_first - state.sim_days_last) / (state.sim_purchase_days - 1)
        overdue_ratio = state.sim_days_last / average_gap
    else:
        average_gap, overdue_ratio = np.nan, np.nan

    row = pd.Series({
        "InvoiceCountLast365Days": state.sim_invoices_365,
        "PurchaseAmountLast365Days": state.sim_amount_365,
        "DistinctProductCountLast365Days": state.sim_products_365,
        "DaysSinceLastPurchase": state.sim_days_last,
        "DaysSinceFirstPurchase": state.sim_days_first,
        "AverageDaysBetweenPurchases": average_gap,
        "OverdueRatio": overdue_ratio,
        "InvoiceCountLast90Days": state.sim_invoices_90,
        "PurchaseAmountLast90Days": state.sim_amount_90,
        "Country": state.sim_country,
    })
    probability = float(churn_probability(load_model(), row.to_frame().T)[0])

    show_verdict(probability)
    with st.container(border=True):
        show_reasons(row)

    if name and all(state[key] == value for key, value in example_values(name).items()):
        real = example_rows().loc[EXAMPLES[name]["customer"]]
        happened = "stopped buying" if real["Churned"] == 1 else "ordered again"
        st.caption(f"In reality, this customer {happened} in the next 90 days.")
