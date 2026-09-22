from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"

st.set_page_config(
    page_title="Personal Finance Dashboard",
    page_icon="💰",
    layout="wide",
)

st.title("Personal Finance Dashboard")
st.caption("Income and expense analysis")

monthly = pd.read_csv(REPORTS_DIR / "monthly_summary.csv")
category = pd.read_csv(REPORTS_DIR / "category_summary.csv")

total_income = monthly["Income"].sum()
total_expenses = monthly["Expense"].sum()
net_amount = monthly["Net_INR"].sum()

col1, col2, col3 = st.columns(3)

col1.metric("Total income", f"₹{total_income:,.2f}")
col2.metric("Total expenses", f"₹{total_expenses:,.2f}")
col3.metric("Net amount", f"₹{net_amount:,.2f}")

st.subheader("Monthly cash flow")

monthly_chart = monthly.set_index("Month")[["Income", "Expense"]]
st.line_chart(monthly_chart)

st.subheader("Spending by category")

category_chart = (
    category
    .set_index("Category")[["Total_INR"]]
    .sort_values("Total_INR")
)

st.bar_chart(category_chart, horizontal=True)

st.subheader("Category summary")

st.dataframe(
    category.style.format({
        "Total_INR": "₹{:,.2f}",
        "Average_INR": "₹{:,.2f}",
        "Percentage_of_Expenses": "{:.2f}%",
    }),
    use_container_width=True,
)