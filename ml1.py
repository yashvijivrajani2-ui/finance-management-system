import pandas as pd

df = pd.read_csv("C:/Users/Lenovo/OneDrive/Desktop/projects/finance managment system/transactions_labeled.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())

print(df["Category"].value_counts())

print(df["Category"].nunique())
print(df["Note"].head(20))
print(df["Note"].isnull().sum())
print(df["Note"].astype(str).str.len().describe())
print(df[["INR", "Amount", "Currency", "Income/Expense"]].head(20))
print(df["Income/Expense"].value_counts())
# print(df["Date"].head())
# print(df["Date"].dtype)
df["Date"] = pd.to_datetime(df["Date"], format="mixed")
print("Earliest date:", df["Date"].min())
print("Latest date:", df["Date"].max())

print("\nTransactions per month:")
print(df["Date"].dt.to_period("M").value_counts().sort_index())

print("\nCategory vs Income/Expense:")
print(pd.crosstab(df["Category"], df["Income/Expense"]))
print(df["Label_Source"].value_counts())
print(df.groupby("Income/Expense")["Category"].value_counts())

# Keep only expense transactions
expense_df = df[df["Income/Expense"] == "Expense"].copy()

print("Expense transactions:", len(expense_df))

print("\nExpense categories:")
print(expense_df["Category"].value_counts())

category_counts = expense_df["Category"].value_counts()

print("\nCategories with fewer than 5 records:")
print(category_counts[category_counts < 5])

print("\nMissing notes:", expense_df["Note"].isna().sum())

print("\nExample expense notes:")
print(expense_df[["Note", "Category"]].head(20))


import re

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z\s]", "", s)
    return s.strip()

expense_df["clean_note"] = expense_df["Note"].apply(clean_text)

print(expense_df[["Note", "clean_note", "Category"]].head(20))

print("Empty cleaned notes:", (expense_df["clean_note"] == "").sum())
import re

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z\s]", "", s)
    return s.strip()

expense_df["clean_note"] = expense_df["Note"].apply(clean_text)

print(expense_df[["Note", "clean_note", "Category"]].head(20))
print("Empty cleaned notes:", (expense_df["clean_note"] == "").sum())

import re

def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z\s]", "", s)
    return s.strip()

expense_df["clean_note"] = expense_df["Note"].apply(clean_text)
print("Missing/empty cleaned notes:", (expense_df["clean_note"] == "").sum())

ml_df = expense_df[expense_df["clean_note"] != ""].copy()

print("Rows available for ML:", len(ml_df))
print("\nCategory counts:")
print(ml_df["Category"].value_counts())

# Categories with enough data for classification
category_counts = ml_df["Category"].value_counts()

valid_categories = category_counts[category_counts >= 5].index

train_df = ml_df[ml_df["Category"].isin(valid_categories)].copy()

print("Rows for classifier:", len(train_df))

print("\nCategories used for training:")
print(train_df["Category"].value_counts())

import pandas as pd

# Load original dataset
df = pd.read_csv("transactions_labeled.csv")

print("Original dataset shape:", df.shape)
print(df.head())

# Keep only expense transactions
expense_df = df[df["Income/Expense"] == "Expense"].copy()

print("Expense transactions:", len(expense_df))

import re

def clean_text(s):
    if pd.isna(s):
        return ""
    
    s = str(s).lower()
    s = re.sub(r"[^a-z\s]", "", s)
    return s.strip()

expense_df["clean_note"] = expense_df["Note"].apply(clean_text)

# Remove rows where there is no transaction description
ml_df = expense_df[expense_df["clean_note"] != ""].copy()

print("Rows with usable notes:", len(ml_df))

# Count examples for each category
category_counts = ml_df["Category"].value_counts()

# Keep categories with at least 5 examples
valid_categories = category_counts[category_counts >= 5].index

ml_df = ml_df[ml_df["Category"].isin(valid_categories)].copy()

print("Rows for ML training:", len(ml_df))

print("\nCategories:")
print(ml_df["Category"].value_counts())

# Keep only the columns needed for text classification
ml_df = ml_df[["Note", "clean_note", "Category"]]

ml_df.to_csv("ml_training_data.csv", index=False)

print("\nML dataset saved successfully!")
print("File: ml_training_data.csv")

