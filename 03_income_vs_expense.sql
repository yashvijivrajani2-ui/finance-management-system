WITH monthly_income AS (
    SELECT
        strftime('%Y-%m', Date) AS month,
        SUM(Amount) AS income_total
    FROM transactions_labeled
    WHERE "Income/Expense" = 'income'
    GROUP BY month
),
monthly_expense AS (
    SELECT
        strftime('%Y-%m', Date) AS month,
        SUM(Amount) AS expense_total
    FROM transactions_labeled
    WHERE "Income/Expense" = 'expense'
    GROUP BY month
)
SELECT
    i.month,
    i.income_total,
    e.expense_total,
    (i.income_total - e.expense_total) AS net_savings
FROM monthly_income i
JOIN monthly_expense e
    ON i.month = e.month
ORDER BY i.month;