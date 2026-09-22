from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "raw_expenses.csv"

st.set_page_config(
    page_title="Personal Finance Dashboard",
    page_icon="💰",
    layout="wide",
)


@st.cache_data
def load_data():
    df = pd.read_csv(RAW_FILE)

    empty_columns = df.columns[df.isna().all()].tolist()
    df = df.drop(columns=empty_columns).copy()

    df["Date"] = pd.to_datetime(
        df["Date"].astype("string").str.strip(),
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )

    df["Amount_INR"] = df["INR"]

    return df


df = load_data()

st.title("Personal Finance Dashboard")
st.caption("Income and expense analysis")

st.sidebar.header("Filters")

categories = sorted(df["Category"].dropna().unique().tolist())

selected_categories = st.sidebar.multiselect(
    "Category",
    options=categories,
    default=categories,
)

transaction_types = st.sidebar.multiselect(
    "Transaction type",
    options=["Income", "Expense"],
    default=["Income", "Expense"],
)

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

selected_dates = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

filtered_df = df[
    df["Category"].isin(selected_categories)
    & df["Income/Expense"].isin(transaction_types)
].copy()

if len(selected_dates) == 2:
    start_date, end_date = selected_dates
    filtered_df = filtered_df[
        filtered_df["Date"].dt.date.between(start_date, end_date)
    ]

income_total = filtered_df.loc[
    filtered_df["Income/Expense"] == "Income",
    "Amount_INR",
].sum()

expense_total = filtered_df.loc[
    filtered_df["Income/Expense"] == "Expense",
    "Amount_INR",
].sum()

net_amount = income_total - expense_total

col1, col2, col3, col4 = st.columns(4)

col1.metric("Income", f"₹{income_total:,.2f}")
col2.metric("Expenses", f"₹{expense_total:,.2f}")
net_label = "Surplus" if net_amount >= 0 else "Deficit"

col3.metric(
    net_label,
    f"₹{net_amount:,.2f}",
)
col4.metric("Transactions", f"{len(filtered_df):,}")

if net_amount < 0:
    st.warning(
        f"Your recorded expenses exceed your recorded income by "
        f"₹{abs(net_amount):,.2f}."
    )
else:
    st.success(
        f"Your recorded income exceeds your expenses by "
        f"₹{net_amount:,.2f}."
    )

st.subheader("Monthly cash flow")

monthly = (
    filtered_df.assign(
        Month=filtered_df["Date"].dt.to_period("M").astype(str)
    )
    .groupby(["Month", "Income/Expense"], as_index=False)
    .agg(Total_INR=("Amount_INR", "sum"))
    .pivot(
        index="Month",
        columns="Income/Expense",
        values="Total_INR",
    )
    .fillna(0)
)

for column in ["Income", "Expense"]:
    if column not in monthly.columns:
        monthly[column] = 0.0

st.line_chart(monthly[["Income", "Expense"]])

st.subheader("Spending by category")

category_summary = (
    filtered_df[filtered_df["Income/Expense"] == "Expense"]
    .groupby("Category", as_index=False)
    .agg(
        Total_INR=("Amount_INR", "sum"),
        Transactions=("Amount_INR", "count"),
    )
    .sort_values("Total_INR", ascending=False)
)

st.bar_chart(
    category_summary.set_index("Category")[["Total_INR"]]
)

st.subheader("Transactions")

display_columns = [
    "Date",
    "Category",
    "Income/Expense",
    "Amount",
    "Currency",
    "Amount_INR",
]

st.dataframe(
    filtered_df[display_columns].sort_values("Date", ascending=False),
    use_container_width=True,
)