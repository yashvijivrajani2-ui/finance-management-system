from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from forecast import forecast_category
from ml_predict import clean_text, model, predict_category, vectorizer


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

    if df["Date"].isna().any():
        raise ValueError("Some dates could not be parsed.")

    df["Amount_INR"] = pd.to_numeric(
        df["INR"],
        errors="coerce",
    )

    df = df.dropna(subset=["Amount_INR"]).copy()

    return df


def get_prediction_probabilities(note):
    cleaned_note = clean_text(note)

    vectorized_note = vectorizer.transform([cleaned_note])

    probabilities = model.predict_proba(vectorized_note)[0]
    categories = model.classes_

    probability_df = pd.DataFrame(
        {
            "Category": categories,
            "Probability": probabilities,
        }
    ).sort_values(
        "Probability",
        ascending=False,
    )

    return probability_df


def build_monthly_expense_data(dataframe):
    expenses = dataframe[
        dataframe["Income/Expense"] == "Expense"
    ].copy()

    if expenses.empty:
        return pd.Series(dtype=float)

    expenses["Month"] = expenses["Date"].dt.to_period("M")

    monthly_expenses = (
        expenses.groupby("Month")["Amount_INR"]
        .sum()
        .sort_index()
    )

    all_months = pd.period_range(
        start=monthly_expenses.index.min(),
        end=monthly_expenses.index.max(),
        freq="M",
    )

    monthly_expenses = monthly_expenses.reindex(
        all_months,
        fill_value=0,
    )

    return monthly_expenses


def build_category_monthly_data(dataframe, category_name):
    expenses = dataframe[
        (dataframe["Income/Expense"] == "Expense")
        & (dataframe["Category"] == category_name)
    ].copy()

    if expenses.empty:
        return pd.Series(dtype=float)

    expenses["Month"] = expenses["Date"].dt.to_period("M")

    category_monthly = (
        expenses.groupby("Month")["Amount_INR"]
        .sum()
        .sort_index()
    )

    all_months = pd.period_range(
        start=category_monthly.index.min(),
        end=category_monthly.index.max(),
        freq="M",
    )

    category_monthly = category_monthly.reindex(
        all_months,
        fill_value=0,
    )

    return category_monthly


def generate_suggestions(dataframe, income, expenses):
    suggestions = []

    expense_data = dataframe[
        dataframe["Income/Expense"] == "Expense"
    ].copy()

    net_amount = income - expenses

    if net_amount < 0:
        suggestions.append(
            f"Your expenses are ₹{abs(net_amount):,.2f} higher "
            "than your income for the selected period."
        )
    else:
        suggestions.append(
            f"You have a surplus of ₹{net_amount:,.2f} for the "
            "selected period."
        )

    if not expense_data.empty:
        category_totals = (
            expense_data.groupby("Category")["Amount_INR"]
            .sum()
            .sort_values(ascending=False)
        )

        highest_category = category_totals.index[0]
        highest_amount = category_totals.iloc[0]

        suggestions.append(
            f"Your highest spending category is "
            f"{highest_category} at ₹{highest_amount:,.2f}."
        )

        if expenses > 0:
            category_percentage = (
                highest_amount / expenses
            ) * 100

            suggestions.append(
                f"{highest_category} accounts for "
                f"{category_percentage:.1f}% of your total expenses."
            )

    return suggestions


df = load_data()


st.title("Personal Finance Dashboard")
st.caption("Income, expense, and spending analysis")


st.sidebar.header("Filters")


categories = sorted(
    df["Category"].dropna().unique().tolist()
)


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
        filtered_df["Date"].dt.date.between(
            start_date,
            end_date,
        )
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

col3.metric(net_label, f"₹{net_amount:,.2f}")
col4.metric("Transactions", f"{len(filtered_df):,}")


if net_amount < 0:
    st.warning(
        f"Your recorded expenses exceed your income by "
        f"₹{abs(net_amount):,.2f}."
    )
else:
    st.success(
        f"Your recorded income exceeds your expenses by "
        f"₹{net_amount:,.2f}."
    )


st.subheader("Monthly Cash Flow")


monthly = (
    filtered_df.assign(
        Month=filtered_df["Date"].dt.to_period("M").astype(str)
    )
    .groupby(
        ["Month", "Income/Expense"],
        as_index=False,
    )
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


st.subheader("Spending by Category")


category_summary = (
    filtered_df[
        filtered_df["Income/Expense"] == "Expense"
    ]
    .groupby("Category", as_index=False)
    .agg(
        Total_INR=("Amount_INR", "sum"),
        Transactions=("Amount_INR", "count"),
    )
    .sort_values("Total_INR", ascending=False)
)


if category_summary.empty:
    st.info("No expense data matches the selected filters.")
else:
    st.bar_chart(
        category_summary.set_index("Category")[["Total_INR"]],
        horizontal=True,
    )


st.subheader("Expense Distribution")


if not category_summary.empty:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.pie(
        category_summary["Total_INR"],
        labels=category_summary["Category"],
        autopct="%1.1f%%",
    )

    ax.set_title("Expense Distribution")
    st.pyplot(fig)
    plt.close(fig)


st.subheader("Transactions")


display_columns = [
    "Date",
    "Category",
    "Income/Expense",
    "Amount",
    "Currency",
    "Amount_INR",
]


available_display_columns = [
    column
    for column in display_columns
    if column in filtered_df.columns
]


st.dataframe(
    filtered_df[available_display_columns].sort_values(
        "Date",
        ascending=False,
    ),
    width="stretch",
)


st.divider()


st.header("AI-Powered Financial Insights")
st.caption(
    "Category classification, spending forecasts, and "
    "personalized financial suggestions."
)


st.subheader("Expense Category Prediction")


st.write(
    "Enter a transaction description and the trained machine-learning "
    "model will predict its category."
)


prediction_column, amount_column = st.columns(2)


with prediction_column:
    transaction_note = st.text_input(
        "Transaction description",
        placeholder="Example: Uber ride to college",
    )


with amount_column:
    transaction_amount = st.number_input(
        "Transaction amount (₹)",
        min_value=0.0,
        step=10.0,
        value=0.0,
    )


if st.button("Predict Expense Category", type="primary"):
    if not transaction_note.strip():
        st.warning("Please enter a transaction description.")
    else:
        predicted_category, confidence = predict_category(
            transaction_note
        )

        st.success(
    f"Predicted Category: {predicted_category}"
)

if transaction_amount > 0:
    st.info(
        f"Suggested entry: ₹{transaction_amount:,.2f} "
        f"under {predicted_category}."
    )


monthly_expenses = build_monthly_expense_data(filtered_df)


if len(monthly_expenses) >= 6:
    try:
        overall_forecast = forecast_category(
            monthly_expenses,
            "Overall Expenses",
        )

        forecast_value = overall_forecast["forecast"]

        if forecast_value is not None:
            st.subheader("Next-Month Expense Forecast")

            st.metric(
                "Predicted Expense for Next Month",
                f"₹{forecast_value:,.2f}",
            )

            average_monthly_expense = monthly_expenses.mean()

            difference = forecast_value - average_monthly_expense

            if difference > 0:
                st.info(
                    f"The predicted expense is "
                    f"₹{difference:,.2f} above your average "
                    "monthly expense."
                )
            else:
                st.info(
                    f"The predicted expense is "
                    f"₹{abs(difference):,.2f} below your average "
                    "monthly expense."
                )

    except Exception:
        pass


expense_categories = sorted(
    filtered_df.loc[
        filtered_df["Income/Expense"] == "Expense",
        "Category",
    ]
    .dropna()
    .unique()
    .tolist()
)


eligible_forecast_categories = []


for category in expense_categories:
    category_history = build_category_monthly_data(
        filtered_df,
        category,
    )

    if len(category_history) >= 6:
        eligible_forecast_categories.append(category)


if eligible_forecast_categories:
    st.subheader("Category-Wise Expense Forecast")

    selected_forecast_category = st.selectbox(
        "Choose a category",
        options=eligible_forecast_categories,
    )

    category_monthly_expenses = build_category_monthly_data(
        filtered_df,
        selected_forecast_category,
    )

    try:
        category_forecast = forecast_category(
            category_monthly_expenses,
            selected_forecast_category,
        )

        category_forecast_value = category_forecast["forecast"]

        if category_forecast_value is not None:
            st.metric(
                f"Predicted Next-Month "
                f"{selected_forecast_category} Expense",
                f"₹{category_forecast_value:,.2f}",
            )

    except Exception:
        pass


st.subheader("Smart Spending Suggestions")


suggestions = generate_suggestions(
    filtered_df,
    income_total,
    expense_total,
)


for suggestion in suggestions:
    st.info(suggestion)