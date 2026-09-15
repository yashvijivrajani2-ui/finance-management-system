import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


st.set_page_config(
    page_title="AI Personal Finance Advisor",
    layout="wide"
)

st.title("AI-Based Personal Finance Advisor")
st.write("Track your expenses, analyze your spending and get personalized financial recommendations.")


st.sidebar.header("Financial Information")

income = st.sidebar.number_input(
    "Monthly Income (₹)",
    min_value=0.0,
    value=30000.0,
    step=1000.0
)

savings_goal = st.sidebar.number_input(
    "Expected Monthly Savings (₹)",
    min_value=0.0,
    value=8000.0,
    step=500.0
)


st.sidebar.header("Add Expense")

category = st.sidebar.selectbox(
    "Expense Category",
    [
        "Food",
        "Transportation",
        "Education",
        "Shopping",
        "Healthcare",
        "Entertainment",
        "Bills",
        "Rent",
        "Other"
    ]
)

amount = st.sidebar.number_input(
    "Amount (₹)",
    min_value=0.0,
    value=0.0,
    step=100.0
)

if "expenses" not in st.session_state:
    st.session_state.expenses = []


if st.sidebar.button("Add Expense"):

    if amount > 0:

        st.session_state.expenses.append({
            "Category": category,
            "Amount": amount
        })

        st.sidebar.success("Expense added successfully!")

    else:

        st.sidebar.warning("Enter an amount greater than 0.")


if len(st.session_state.expenses) > 0:

    df = pd.DataFrame(st.session_state.expenses)

else:

    df = pd.DataFrame(
        columns=["Category", "Amount"]
    )


total_expenses = df["Amount"].sum()

current_savings = income - total_expenses

if income > 0:
    savings_percentage = (current_savings / income) * 100
else:
    savings_percentage = 0


st.header("Financial Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Monthly Income",
        f"₹{income:,.0f}"
    )

with col2:
    st.metric(
        "Total Expenses",
        f"₹{total_expenses:,.0f}"
    )

with col3:
    st.metric(
        "Current Savings",
        f"₹{current_savings:,.0f}"
    )

with col4:
    st.metric(
        "Savings Goal",
        f"₹{savings_goal:,.0f}"
    )


st.header(" Savings Analysis")

if current_savings >= savings_goal:

    st.success(
        f" Congratulations! You have achieved your savings goal. "
        f"You saved ₹{current_savings:,.0f}."
    )

else:

    required = savings_goal - current_savings

    st.warning(
        f" You need to save ₹{required:,.0f} more "
        f"to reach your goal."
    )


if not df.empty:

    st.header("Expense Analysis")

    category_expenses = (
        df.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Category-wise Spending")

        fig, ax = plt.subplots()

        ax.pie(
            category_expenses.values,
            labels=category_expenses.index,
            autopct="%1.1f%%"
        )

        ax.set_title("Expense Distribution")

        st.pyplot(fig)


    with col2:

        st.subheader("Spending by Category")

        st.bar_chart(category_expenses)


    highest_category = category_expenses.idxmax()
    highest_amount = category_expenses.max()

    st.info(
        f" Your highest spending category is "
        f"*{highest_category}* with "
        f"₹{highest_amount:,.0f} spent."
    )


    st.header(" Personalized Financial Recommendations")

    recommendations = []

    percentage = (
        highest_amount / total_expenses
    ) * 100

    if percentage > 30:

        if highest_category == "Healthcare":

            recommendations.append(
            f"Your healthcare expenses represent "
            f"{percentage:.1f}% of your total expenses. "
            f"Review recurring or non-urgent costs where possible, "
            f"but avoid reducing necessary healthcare."
        )

        elif highest_category == "Rent":

            recommendations.append(
            f"Your rent represents {percentage:.1f}% of your total expenses. "
            f"Consider this fixed cost carefully when planning your budget."
        )

        elif highest_category == "Bills":

            recommendations.append(
            f"Your bills represent {percentage:.1f}% of your total expenses. "
            f"Review recurring bills and reduce unnecessary subscriptions "
            f"or usage where possible."
        )

        elif highest_category == "Shopping":

            recommendations.append(
            f"Your shopping expenses represent {percentage:.1f}% of your "
            f"total expenses. Consider reducing non-essential purchases."
        )

        elif highest_category == "Entertainment":

            recommendations.append(
            f"Your entertainment expenses represent {percentage:.1f}% of "
            f"your total expenses. Consider setting a fixed entertainment budget."
        )

        elif highest_category == "Transportation":

            recommendations.append(
            f"Your transportation expenses represent {percentage:.1f}% of "
            f"your total expenses. Consider more cost-effective travel options "
            f"where practical."
        )

        elif highest_category == "Food":

            recommendations.append(
            f"Your food expenses represent {percentage:.1f}% of your total "
            f"expenses. Review unnecessary or avoidable food purchases."
        )

    else:

        recommendations.append(
            f"Your {highest_category} expenses represent {percentage:.1f}% "
            f"of your total expenses. Review this category and identify "
            f"avoidable expenses before adjusting your budget."
        )

    if current_savings < savings_goal:

        recommendations.append(
            f"You are below your savings goal by "
            f"₹{savings_goal - current_savings:,.0f}. "
            f"Try reducing unnecessary expenses."
        )

    if total_expenses > income:

        recommendations.append(
            " Your expenses are higher than your income. "
            "You should immediately review your spending."
        )

    if current_savings >= savings_goal:

        recommendations.append(
            " Your current savings are on track. "
            "Continue maintaining your spending pattern."
        )

    for recommendation in recommendations:

        st.write(recommendation)



    st.header(" Budget Utilization")

    budget = income - savings_goal

    if budget > 0:

        utilization = (
            total_expenses / budget
        ) * 100

        st.progress(
            min(int(utilization), 100)
        )

        st.write(
            f"You have used *{utilization:.1f}%* "
            f"of your available spending budget."
        )

        if utilization >= 100:

            st.error(
                "🚨 You have exceeded your planned spending budget!"
            )

        elif utilization >= 80:

            st.warning(
                " You are approaching your spending limit."
            )

        else:

            st.success(
                "Your spending is within the planned budget."
            )



    st.header(" AI Expense Forecast")

    st.write(
        "Linear Regression is used as a baseline model "
        "to predict future expenses."
    )

    # Sample historical data for demonstration
    historical_data = pd.DataFrame({
        "Month": [1, 2, 3, 4, 5, 6],
        "Expense": [
            22000,
            24500,
            21000,
            27000,
            23500,
            25000
        ]
    })

    X = historical_data[["Month"]]
    y = historical_data["Expense"]

    model = LinearRegression()

    model.fit(X, y)

    next_month = [[7]]

    predicted_expense = model.predict(next_month)[0]

    st.metric(
        "Predicted Next Month Expense",
        f"₹{predicted_expense:,.0f}"
    )


    prediction_values = model.predict(X)

    fig2, ax2 = plt.subplots()

    ax2.plot(
        historical_data["Month"],
        historical_data["Expense"],
        marker="o",
        label="Actual Expense"
    )

    ax2.plot(
        historical_data["Month"],
        prediction_values,
        linestyle="--",
        label="Predicted Expense"
    )

    ax2.set_xlabel("Month")
    ax2.set_ylabel("Expense (₹)")
    ax2.set_title("Expense Forecasting")

    ax2.legend()

    st.pyplot(fig2)

st.header(" Expense Transactions")

if not df.empty:

    st.dataframe(
        df,
        use_container_width=True
    )

else:

    st.info(
        "No expenses added yet. "
        "Use the sidebar to add your expenses."
    )



st.markdown("---")

st.caption(
    "AI-Based Smart Personal Finance Management System | "
    "Academic Prototype"
)