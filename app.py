import streamlit as st

st.set_page_config(page_title="Customer churn radar", page_icon=":material/redeem:", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/story.py", title="Overview", icon=":material/home:", default=True),
        st.Page("app_pages/risk_list.py", title="At-risk customers", icon=":material/notification_important:"),
        st.Page("app_pages/customer.py", title="Customer lookup", icon=":material/person_search:"),
        st.Page("app_pages/simulator.py", title="What-if", icon=":material/tune:"),
    ],
    position="top",
)

page.run()
