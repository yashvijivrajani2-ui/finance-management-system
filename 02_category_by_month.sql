SELECT
    strftime('%Y-%m', Date) AS month,
    Category,
    SUM(Amount) AS total
FROM transactions_labeled
WHERE "Income/Expense" = 'expense'
GROUP BY month, Category
ORDER BY month, total DESC;