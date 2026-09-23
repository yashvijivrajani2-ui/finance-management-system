# ML Module Handoff

## 1. Category Classification

### Purpose

Predict the expense category from transaction note/text.

### Input

A transaction note, for example:

```text
"Pizza"
"Bus ticket"
"Rent"
```

### Output

The model returns:

* Predicted category
* Prediction confidence

Example:

```text
Transportation
0.44
```

### Model

* TF-IDF Vectorizer
* Logistic Regression
* Class-balanced training using `class_weight="balanced"`

### Training Data

* Expense transactions only
* 219 usable transaction notes
* Categories used for classification:

  * Food
  * Transportation
  * Other
  * Apparel
  * Household

### Performance

Overall test accuracy:

**93.18%**

### Strong Categories

* Food
* Transportation

### Weak Categories

* Apparel
* Household
* Other

### Limitation

The dataset is relatively small and highly imbalanced. Some categories contain very few examples, so their classification performance is less reliable.

The model should be retrained when more labeled transaction data becomes available.

---

## 2. Prediction Interface

Person 4 can use the trained classifier through:

```python
from ml_predict import predict_category

category, confidence = predict_category("Pizza")
```

The function returns:

```text
category
confidence
```

Saved model files:

```text
models/
├── category_classifier.joblib
└── vectorizer.joblib
```

---

## 3. Spending Forecasting

### Purpose

Forecast future spending using historical monthly transaction amounts.

### Model

ARIMA `(1,1,1)`

### Input

Monthly spending data grouped by category.

### Current Status

Forecast is currently **not generated**.

### Reason

The dataset contains only **5 months** of historical data.

The forecasting function requires at least **6 months** of historical data.

Example output:

```text
{
    'category': 'Transportation',
    'forecast': None,
    'status': 'Insufficient historical data',
    'months_available': 5
}
```

### Interface

Person 4 can use:

```python
from forecast import get_forecast

result = get_forecast("Transportation", category_monthly)
```

### Reliability

Forecasting should not be relied upon until sufficient historical data is available.

---

## 4. Suggestion Engine

### Purpose

Generate spending suggestions using actual transaction data, financial goals, and forecast availability.

### Input

```text
month_data
goals
forecast
```

`month_data` contains:

```text
income
expense
by_category
```

### Output

A list of text-based spending suggestions.

Example:

```text
Your highest spending category is Other at ₹300.
Spending forecast is unavailable because there is not enough historical data.
```

### Function

```python
generate_suggestions(month_data, goals, forecast)
```

### Current Status

Working and tested using the actual transaction dataset.

---

## 5. Important Files

```text
finance managment system/
│
├── m2.py
├── ml_predict.py
├── forecast.py
├── ML_HANDOFF.md
│
└── models/
    ├── category_classifier.joblib
    └── vectorizer.joblib
```

### File Responsibilities

| File                         | Purpose                                               |
| ---------------------------- | ----------------------------------------------------- |
| `m2.py`                      | Main ML development, training, testing and evaluation |
| `ml_predict.py`              | Reusable transaction-category prediction interface    |
| `forecast.py`                | Spending forecast functions                           |
| `ML_HANDOFF.md`              | Documentation for integration                         |
| `category_classifier.joblib` | Trained Logistic Regression model                     |
| `vectorizer.joblib`          | Trained TF-IDF vectorizer                             |

---

## 6. Important Limitations

1. The classification dataset is relatively small.
2. The classification dataset is imbalanced.
3. Some categories have very few labeled examples.
4. Forecasting is currently unavailable because only 5 months of historical data are available.
5. At least 6 months of historical data are required by the current forecasting function.
6. More transaction data should be collected for more reliable forecasting.
7. Current category labels are manually assigned.
8. The classifier should be retrained when sufficient new labeled data becomes available.

---

## 7. Integration Summary for Person 4

### Category Prediction

Use:

```python
from ml_predict import predict_category

category, confidence = predict_category(transaction_note)
```

### Spending Forecast

Use:

```python
from forecast import get_forecast

forecast_result = get_forecast(category_name, category_monthly)
```

### Suggestions

Use:

```python
suggestions = generate_suggestions(
    month_data,
    goals,
    forecast
)
```

The ML module is designed so that Person 4 can use these interfaces without needing to modify the underlying machine-learning implementation.
