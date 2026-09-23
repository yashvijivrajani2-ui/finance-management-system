# person 4
import joblib
import re


# Load trained model and vectorizer
model = joblib.load("models/category_classifier.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")


def clean_text(s):

    if s is None:
        return ""

    s = str(s).lower()
    s = re.sub(r"[^a-z\s]", "", s)

    return s.strip()


def predict_category(note):

    cleaned_note = clean_text(note)

    vectorized_note = vectorizer.transform([cleaned_note])

    predicted_category = model.predict(vectorized_note)[0]

    confidence = model.predict_proba(vectorized_note).max()

    return predicted_category, confidence
