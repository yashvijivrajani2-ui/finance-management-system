import pandas as pd
import re
from forecast import forecast_category
from forecast import get_forecast
from ml_predict import predict_category

def clean_text(s):
    if pd.isna(s):
        return ""
    
    s = str(s).lower()
    s = re.sub(r"[^a-z\s]", "", s)
    return s.strip()

# Load ML training dataset
df = pd.read_csv("ml_training_data.csv")
print("\nCategories used for training:")
print(df["Category"].value_counts())

print("Dataset shape:", df.shape)
print(df.head())

X = df["clean_note"]
y = df["Category"]

print("\nInput column:", X.name)
print("Target column:", y.name)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\nTraining feature shape:", X_train_vec.shape)
print("Testing feature shape:", X_test_vec.shape)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)

print("\nModel training completed!")

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
import os
import joblib

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save trained model
joblib.dump(model, "models/category_classifier.joblib")

# Save TF-IDF vectorizer
joblib.dump(vectorizer, "models/vectorizer.joblib")

print("\nModel and vectorizer saved successfully!")

# PART 2 — SPENDING FORECASTING

# Load original transaction dataset
original_df = pd.read_csv("transactions_labeled.csv")

print("\nOriginal dataset loaded for forecasting:", original_df.shape)
original_df["Date"] = pd.to_datetime(
    original_df["Date"],
    format="mixed"
)

print("Date conversion completed.")

print("Earliest date:", original_df["Date"].min())
print("Latest date:", original_df["Date"].max())

forecast_expenses = original_df[
    original_df["Income/Expense"] == "Expense"
].copy()

print("\nExpense transactions for forecasting:", len(forecast_expenses))

monthly = (
    forecast_expenses
    .groupby(forecast_expenses["Date"].dt.to_period("M"))["Amount"]
    .sum()
    .reset_index()
)

monthly.columns = ["month", "total"]

monthly["month"] = monthly["month"].astype(str)

print("\nMonthly spending:")
print(monthly)

print("\nNumber of months available:", len(monthly))

if len(monthly) < 6:
    print("WARNING: Limited historical data. Forecast reliability is low.")

forecast = forecast_category(monthly["total"], "Overall")

if forecast is not None:
    print("\nForecast for next month:")
    print(forecast)
else:
    print("\nForecast could not be generated reliably.")

# CHECK CATEGORY-WISE MONTHLY HISTORY

category_monthly = (
    forecast_expenses
    .groupby([
        forecast_expenses["Date"].dt.to_period("M"),
        "Category"
    ])["Amount"]
    .sum()
    .reset_index()
)

category_monthly.columns = ["month", "category", "total"]
category_monthly["month"] = category_monthly["month"].astype(str)

print("\nCategory-wise monthly spending:")
print(category_monthly)

print("\nNumber of months available for each category:")

category_history = (
    category_monthly
    .groupby("category")["month"]
    .nunique()
    .sort_values(ascending=False)
)

print(category_history)

forecast_results = []

for category in category_history.index:

    category_data = category_monthly[
        category_monthly["category"] == category
    ]["total"]

    result = forecast_category(
        category_data,
        category
    )

    forecast_results.append(result)

forecast_results_df = pd.DataFrame(forecast_results)

print("SPENDING FORECAST RESULTS")

for _, row in forecast_results_df.iterrows():

    if row["forecast"] is not None:
        print(
            f"{row['category']}: "
            f"₹{row['forecast']:.2f} forecast "
            f"({row['months_available']} months available)"
        )

    else:
        print(
            f"{row['category']}: "
            f"Forecast unavailable — "
            f"only {row['months_available']} months available"
        )

# PART 3 — SUGGESTION ENGINE

def generate_suggestions(month_data, goals, forecast=None):

    suggestions = []

    income = month_data["income"]
    expense = month_data["expense"]
    by_category = month_data["by_category"]

    net = income - expense

    # 1. Check whether spending is greater than income
    if net < 0:
        suggestions.append(
            "Your expenses are higher than your income this month."
        )

    # 2. Find the highest spending category
    if not by_category.empty:

        top_category = by_category.idxmax()
        top_amount = by_category.max()

        suggestions.append(
            f"Your highest spending category is {top_category} "
            f"at ₹{top_amount:.0f}."
        )

    # 3. Goal-based suggestion
    if goals:

        required = sum(
            (goal["target"] - goal["saved"])
            / max(1, goal["months_left"])
            for goal in goals
        )

        if net < required:

            suggestions.append(
                f"You need to save approximately ₹{required:.0f} "
                "per month to stay on track with your goals."
            )

    # 4. Forecast warning
    if forecast is None:

        suggestions.append(
            "Spending forecast is unavailable because there is "
            "not enough historical data."
        )

    return suggestions


# GENERATE SUGGESTIONS FROM REAL DATA

latest_month = original_df["Date"].dt.to_period("M").max()

latest_data = original_df[
    original_df["Date"].dt.to_period("M") == latest_month
]

income = latest_data[
    latest_data["Income/Expense"] == "Income"
]["Amount"].sum()

expense = latest_data[
    latest_data["Income/Expense"] == "Expense"
]["Amount"].sum()

by_category = latest_data[
    latest_data["Income/Expense"] == "Expense"
].groupby("Category")["Amount"].sum()

month_data = {
    "income": income,
    "expense": expense,
    "by_category": by_category
}

goals = []

suggestions = generate_suggestions(
    month_data,
    goals,
    forecast=None
)

print("\nReal Data Suggestions:")
print("Month:", latest_month)
print("Income:", income)
print("Expense:", expense)

for suggestion in suggestions:
    print("-", suggestion)

# ML EVALUATION SUMMARY

print("ML EVALUATION SUMMARY")

print("\n1. CATEGORY CLASSIFIER")
print(f"Accuracy: {accuracy:.2%}")

print("\nStrongest categories:")
print("- Food: Strong performance")
print("- Transportation: Good performance")

print("\nWeak categories:")
print("- Apparel")
print("- Household")
print("- Other")

print("\nReason for weak performance:")
print("These categories have very few training examples,")
print("so the model has limited information to learn from.")

print("\n2. SPENDING FORECASTING")
print("Historical months available: 5")
print("Minimum required by the forecasting function: 6")
print("Forecast status: Not generated")
print("Reliability: Low due to limited historical data")

print("\n3. SUGGESTION ENGINE")
print("Status: Working")
print("Suggestions are generated using actual transaction data.")

print("END OF ML EVALUATION")

# FUNCTION FOR DASHBOARD INTEGRATION
# necessary for person 4

def predict_category(note):

    cleaned_note = clean_text(note)

    vectorized_note = vectorizer.transform([cleaned_note])

    predicted_category = model.predict(vectorized_note)[0]

    confidence = model.predict_proba(vectorized_note).max()

    return predicted_category, confidence

def show_probabilities(note):

    cleaned_note = clean_text(note)

    vectorized_note = vectorizer.transform([cleaned_note])

    probabilities = model.predict_proba(vectorized_note)[0]
    categories = model.classes_

    results = sorted(
        zip(categories, probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\nProbabilities for: {note}")

    for category, probability in results:
        print(f"{category}: {probability:.2f}")

# Test prediction

test_notes = [
    "Pizza",
    "Bus ticket",
    "Metro",
    "Auto ride",
    "Restaurant",
    "Shoe",
    "Hoodie",
    "Rent"
]

print("\nPrediction Tests:")

for note in test_notes:
    category, confidence = predict_category(note)
    print(f"{note:15} -> {category:15} | Confidence: {confidence:.2f}")

print("\nCategory and note examples:")

for category in sorted(df["Category"].unique()):
    category_notes = df[df["Category"] == category]["Note"].dropna()

    print(f"\n--- {category} ({len(category_notes)} notes) ---")

    for note in category_notes.tolist():
        print(note)

show_probabilities("Bus ticket")
show_probabilities("Shoe")
show_probabilities("Rent")

print("\nSaved model test:")

test_category, test_confidence = predict_category("Uber ride")

print("Uber ride ->", test_category)
print("Confidence:", round(test_confidence, 2))

print("\nTesting saved model with other examples:")

for note in ["Pizza", "Bus ticket", "Metro", "Shoe", "Rent"]:
    category, confidence = predict_category(note)
    print(note, "->", category, round(confidence, 2))

result = get_forecast("Transportation", category_monthly)

print("\nDashboard Forecast Test:")
print(result)

category, confidence = predict_category("Bus ticket")

print("\nFinal ML Interface Test:")
print("Category:", category)
print("Confidence:", round(confidence, 2))