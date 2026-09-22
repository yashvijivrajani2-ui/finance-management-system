from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "raw_expenses.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_and_clean_data(file_path):
    df_raw = pd.read_csv(file_path)

    empty_columns = df_raw.columns[df_raw.isna().all()].tolist()

    df = df_raw.drop(columns=empty_columns).copy()

    df["Date"] = pd.to_datetime(
        df["Date"].astype("string").str.strip(),
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )

    if df["Date"].isna().any():
        raise ValueError("Some dates could not be parsed.")

    df["Amount_INR"] = df["INR"]

    return df


def create_category_summary(df):
    expenses = df[df["Income/Expense"] == "Expense"]

    summary = (
        expenses
        .groupby("Category", as_index=False)
        .agg(
            Total_INR=("Amount_INR", "sum"),
            Transactions=("Amount_INR", "count"),
            Average_INR=("Amount_INR", "mean"),
        )
        .sort_values("Total_INR", ascending=False)
    )

    summary["Percentage_of_Expenses"] = (
        summary["Total_INR"]
        / summary["Total_INR"].sum()
        * 100
    )

    return summary


def create_monthly_summary(df):
    monthly_summary = (
        df.assign(Month=df["Date"].dt.to_period("M").astype(str))
        .groupby(["Month", "Income/Expense"], as_index=False)
        .agg(
            Total_INR=("Amount_INR", "sum"),
            Transactions=("Amount_INR", "count"),
        )
    )

    monthly_pivot = (
        monthly_summary
        .pivot(
            index="Month",
            columns="Income/Expense",
            values="Total_INR",
        )
        .fillna(0)
        .reset_index()
    )

    for column in ["Income", "Expense"]:
        if column not in monthly_pivot.columns:
            monthly_pivot[column] = 0.0

    all_months = pd.period_range(
        start=df["Date"].min().to_period("M"),
        end=df["Date"].max().to_period("M"),
        freq="M",
    ).astype(str)

    monthly_pivot = (
        monthly_pivot
        .set_index("Month")
        .reindex(all_months, fill_value=0)
        .rename_axis("Month")
        .reset_index()
    )

    monthly_pivot["Net_INR"] = (
        monthly_pivot["Income"] - monthly_pivot["Expense"]
    )

    return monthly_pivot


def main():
    REPORTS_DIR.mkdir(exist_ok=True)

    df = load_and_clean_data(RAW_FILE)

    category_summary = create_category_summary(df)
    monthly_summary = create_monthly_summary(df)

    category_summary.to_csv(
        REPORTS_DIR / "category_summary.csv",
        index=False,
    )

    monthly_summary.to_csv(
        REPORTS_DIR / "monthly_summary.csv",
        index=False,
    )

    print(f"Loaded {len(df)} transactions.")
    print("Reports created successfully.")


if __name__ == "__main__":
    main()