from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"


monthly = pd.read_csv(REPORTS_DIR / "monthly_summary.csv")
category = pd.read_csv(REPORTS_DIR / "category_summary.csv")

# Monthly cash flow chart
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    monthly["Month"],
    monthly["Income"],
    marker="o",
    label="Income",
)

ax.plot(
    monthly["Month"],
    monthly["Expense"],
    marker="o",
    label="Expenses",
)

ax.set_title("Monthly Income and Expenses")
ax.set_xlabel("Month")
ax.set_ylabel("Amount (INR)")
ax.tick_params(axis="x", rotation=45)
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(
    REPORTS_DIR / "monthly_cashflow.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)

# Category spending chart
top_categories = category.head(10).sort_values("Total_INR")

fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(
    top_categories["Category"],
    top_categories["Total_INR"],
)

ax.set_title("Top Expense Categories")
ax.set_xlabel("Total spending (INR)")
ax.set_ylabel("Category")

fig.tight_layout()
fig.savefig(
    REPORTS_DIR / "category_spending.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)

print("Charts created successfully.")