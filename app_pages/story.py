import streamlit as st

from utils import LATEST_DATE, TEST, load_purchases, load_training_data, score_snapshot, svg

purchases = load_purchases()
data = load_training_data()

st.image(svg("hero.svg"), width="stretch")

st.title("Keeping wholesale customers from slipping away")
st.markdown(
    "A UK online gift wholesaler sells to thousands of shops and resellers. Customers rarely announce that "
    "they are leaving. **They just stop ordering.** This app spots them early, so the sales team can act "
    "while the customer can still be won back."
)

# ---------- 1 ----------
st.header("1. The business")
with st.container(horizontal=True):
    st.metric("Purchase lines", f"{len(purchases):,}", border=True)
    st.metric("Orders", f"{purchases['Invoice'].nunique():,}", border=True)
    st.metric("Customers", f"{purchases['Customer ID'].nunique():,}", border=True)
    st.metric("Countries", f"{purchases['Country'].nunique()}", border=True)
st.caption(
    f"Cleaned transactions from {purchases['InvoiceDate'].min():%B %Y} to {purchases['InvoiceDate'].max():%B %Y}. "
    "Many customers are wholesalers who order gifts and decorations in bulk."
)

# ---------- 2 ----------
st.header("2. The problem: customers leave quietly")
one_order = data[data["InvoiceCountLast365Days"] == 1]["Churned"].mean()
frequent = data[data["InvoiceCountLast365Days"] > 10]["Churned"].mean()
with st.container(horizontal=True):
    st.metric(
        "Snapshots with no order in the next 90 days", f"{data['Churned'].mean():.0%}",
        help="Share of monthly customer snapshots where the customer made no purchase in the following 90 days.",
        border=True,
    )
    st.metric("Churn rate of one-order customers", f"{one_order:.0%}", border=True)
    st.metric("Churn rate of customers with 11+ orders", f"{frequent:.0%}", border=True)
st.markdown(
    "Most customers who stop buying give no warning. By the time someone notices, the customer has "
    "already found another supplier. Losing a wholesale customer costs all of their future orders, "
    "while a retention call costs a few minutes."
)

# ---------- 3 ----------
st.header("3. The question")
st.markdown(
    "On the 1st of every month, look at what each active customer did over the **past year** and "
    "predict whether they will make **no purchase in the next 90 days**."
)
st.image(svg("timeline.svg"), width="stretch")

# ---------- 4 ----------
st.header("4. The answer")
st.markdown(
    f"A Random Forest was tested on **{TEST['customers']:,} customer snapshots from August and September 2011** "
    "that it had never seen."
)
with st.container(horizontal=True):
    st.metric("Leavers caught (recall)", f"{TEST['recall']:.0%}", border=True)
    st.metric("Flagged who really left (precision)", f"{TEST['precision']:.0%}", border=True)
    st.metric(
        "Accuracy", f"{TEST['accuracy']:.0%}",
        delta=f"{(TEST['accuracy'] - TEST['baseline_accuracy']) * 100:+.0f} pts vs flagging everyone",
        border=True,
    )
with st.container(border=True):
    st.markdown(
        f":material/support_agent: **In plain words:** calling the {TEST['flagged']:,} customers the model flagged would "
        f"have reached **{TEST['caught']:,} of the {TEST['leavers']:,} customers who were about to leave**. About 2 in 3 "
        "calls go to a real leaver; the rest are cheap false alarms."
    )

# ---------- 5 ----------
st.header("5. How to use it")
latest = score_snapshot(LATEST_DATE)
st.markdown(
    f"Today is **{LATEST_DATE.day} {LATEST_DATE:%B %Y}**, the latest date in the data. The app reads the invoice history, "
    f"so nothing needs to be typed in. It currently flags **{(latest['Churn probability'] >= 0.5).sum():,} of "
    f"{len(latest):,} active customers** as likely to stop buying."
)
with st.container(horizontal=True):
    for page, step, title, text, icon in [
        ("app_pages/risk_list.py", "Step 1", "See who is at risk", "Every active customer ranked by churn risk, with a suggested action. Download the list for the sales team.", ":material/notification_important:"),
        ("app_pages/customer.py", "Step 2", "Check one customer", "Before calling, see the customer's risk, the reasons behind it and their recent spending.", ":material/person_search:"),
        ("app_pages/simulator.py", "Step 3", "Try a what-if", "Change a customer's history and see how the risk responds, starting from real examples.", ":material/tune:"),
    ]:
        with st.container(border=True, height="stretch"):
            st.caption(step)
            st.markdown(f"**{title}**")
            st.markdown(text)
            st.page_link(page, label="Open", icon=icon)

st.subheader("How it helps the business")
with st.container(horizontal=True):
    for icon, title, text in [
        (":material/target:", "Focus", "The team spends its time on the customers who are actually slipping away."),
        (":material/timer:", "Act early", "Customers are flagged while they can still be won back, not after they have gone."),
        (":material/savings:", "Protect revenue", "Each list shows how much the at-risk customers spent last year: the business at stake."),
        (":material/event_repeat:", "Simple routine", "Run it on the 1st of every month from the existing invoice data."),
    ]:
        with st.container(border=True, height="stretch"):
            st.markdown(f"{icon} **{title}**")
            st.caption(text)

st.caption("Data: Chen, D. (2012), Online Retail II, UCI Machine Learning Repository, DOI 10.24432/C5CG6D, CC BY 4.0.")
