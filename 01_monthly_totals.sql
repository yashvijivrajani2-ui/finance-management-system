SELECT
    strftime('%Y-%m', Date) AS month,
    SUM(Amount) AS total_expense
FROM transactions_labeled
WHERE "Income/Expense" = 'expense'
GROUP BY month
ORDER BY month;