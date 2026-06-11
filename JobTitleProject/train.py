import os
import re

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "job_title_des.csv")
MODEL_PATH = os.path.join(BASE_DIR, "job_title_classifier.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")

lemmatizer = WordNetLemmatizer()
DEFAULT_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
    "will", "with", "you", "your", "our", "we", "this", "or", "if",
}


def ensure_nltk_data():
    for package, path in [
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(package, quiet=True, raise_on_error=True)
            except Exception:
                pass


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = DEFAULT_STOP_WORDS

    cleaned_tokens = []
    for word in tokens:
        if word in stop_words or len(word) <= 2:
            continue
        try:
            cleaned_tokens.append(lemmatizer.lemmatize(word))
        except LookupError:
            cleaned_tokens.append(word)
    tokens = cleaned_tokens
    return " ".join(tokens)


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError("Place job_title_des.csv in the project root before training.")

    ensure_nltk_data()
    data = pd.read_csv(DATASET_PATH)
    data = data.dropna(subset=["Job Title", "Job Description"])
    data["clean_description"] = data["Job Description"].apply(preprocess_text)

    x_train, x_test, y_train, y_test = train_test_split(
        data["clean_description"],
        data["Job Title"],
        test_size=0.2,
        random_state=42,
        stratify=data["Job Title"] if data["Job Title"].value_counts().min() > 1 else None,
    )

    vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95
)
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    base_model = LinearSVC(
    C=2.0,
    class_weight='balanced',
    random_state=42,
    dual='auto'
)

    classifier = CalibratedClassifierCV(base_model)
    classifier.fit(x_train_vectorized, y_train)
    predictions = classifier.predict(x_test_vectorized)

    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print(classification_report(y_test, predictions, zero_division=0))

    joblib.dump(classifier, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("Saved job_title_classifier.joblib and tfidf_vectorizer.joblib")


if __name__ == "__main__":
    main()
