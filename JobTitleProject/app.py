import os
import re
from datetime import datetime

import joblib
import nltk
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from werkzeug.security import check_password_hash, generate_password_hash

from database import close_db, init_db
from models import (
    create_prediction,
    create_user,
    get_history,
    get_user_by_email,
    get_user_by_id,
    get_user_by_username,
    get_user_stats,
)


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "job_title_classifier.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-this-secret-key-before-deployment")

classifier = None
vectorizer = None
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


def load_ml_artifacts():
    global classifier, vectorizer

    print("MODEL PATH:", MODEL_PATH)
    print("VECTORIZER PATH:", VECTORIZER_PATH)
    print("MODEL EXISTS:", os.path.exists(MODEL_PATH))
    print("VECTORIZER EXISTS:", os.path.exists(VECTORIZER_PATH))

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        classifier = None
        vectorizer = None
        return

    classifier = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("CLASSIFIER TYPE:", type(classifier))
    print("VECTORIZER TYPE:", type(vectorizer))
    print("HAS IDF:", hasattr(vectorizer, "idf_"))
    try:
        print("VOCAB SIZE:", len(vectorizer.vocabulary_))
    except Exception as e:
        print("VOCAB ERROR:", e)


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


def confidence_from_svc(description_vector, predicted_title):
    probabilities = classifier.predict_proba(description_vector)
    return round(float(probabilities.max()) * 100, 2)


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return get_user_by_id(user_id)


def wants_json():
    return request.is_json or request.accept_mimetypes.best == "application/json" or request.args.get("format") == "json"


def login_required(view):
    def wrapped(*args, **kwargs):
        if not current_user():
            if request.path.startswith("/api") or request.is_json:
                return jsonify({"success": False, "message": "Authentication required"}), 401
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    wrapped.__name__ = view.__name__
    return wrapped


def admin_required(view):
    def wrapped(*args, **kwargs):
        user = current_user()
        if not user:
            return redirect(url_for("login"))
        if not user["is_admin"]:
            return redirect(url_for("dashboard"))
        return view(*args, **kwargs)

    wrapped.__name__ = view.__name__
    return wrapped


@app.context_processor
def inject_user():
    return {"current_user": current_user()}


@app.route("/")
def welcome():
    if current_user():
        return redirect(url_for("dashboard"))
    return render_template("welcome.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")

    data = request.get_json(silent=True) or request.form
    username = data.get("username", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not username or not email or not password:
        return respond(False, "All fields are required", "signup.html", 400)
    if len(password) < 6:
        return respond(False, "Password must contain at least 6 characters", "signup.html", 400)
    if get_user_by_username(username) or get_user_by_email(email):
        return respond(False, "Username or email already exists", "signup.html", 409)

    password_hash = generate_password_hash(password)
    create_user(username, email, password_hash)
    return respond(True, "Signup successful. Please login.", "login.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    data = request.get_json(silent=True) or request.form
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    user = get_user_by_email(email)

    if not user or not check_password_hash(user["password_hash"], password):
        return respond(False, "Invalid email or password", "login.html", 401)

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["is_admin"] = bool(user["is_admin"])
    return respond(True, "Login successful", redirect_url=url_for("dashboard"))


@app.route("/logout")
def logout():
    session.clear()
    if wants_json():
        return jsonify({"success": True, "message": "Logged out"})
    return redirect(url_for("welcome"))


@app.route("/dashboard")
@login_required
def dashboard():
    if wants_json():
        user = current_user()
        payload = {"success": True, "user": dict(user)}
        if user["is_admin"]:
            payload["stats"] = get_user_stats()
        return jsonify(payload)
    return render_template("dashboard.html")


@app.route("/history")
@login_required
def history():
    search = request.args.get("search", "").strip()
    rows = get_history(session["user_id"], search)
    if wants_json():
        return jsonify({"success": True, "history": [dict(row) for row in rows]})
    return render_template("history.html", predictions=rows, search=search)


@app.route("/admin")
@admin_required
def admin_dashboard():
    stats = get_user_stats()
    return render_template("admin_dashboard.html", stats=stats)


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if classifier is None or vectorizer is None:
        return jsonify({
            "success": False,
            "message": "ML model files are missing. Place job_title_classifier.joblib and tfidf_vectorizer.joblib in the project root.",
        }), 503

    data = request.get_json(silent=True) or request.form
    description = data.get("job_description", "").strip()
    if len(description) < 20:
        return jsonify({"success": False, "message": "Please enter a longer job description."}), 400

    cleaned = preprocess_text(description)
    description_vector = vectorizer.transform([cleaned])
    predicted_title = classifier.predict(description_vector)[0]
    confidence = confidence_from_svc(description_vector, predicted_title)
    create_prediction(session["user_id"], description, predicted_title)

    return jsonify({
        "success": True,
        "predicted_job_title": predicted_title,
        "confidence_score": confidence,
    })


@app.route("/api/signup", methods=["POST"])
def api_signup():
    return signup()


@app.route("/api/login", methods=["POST"])
def api_login():
    return login()


@app.route("/api/logout")
def api_logout():
    session.clear()
    return jsonify({"success": True, "message": "Logged out"})


@app.route("/api/history")
@login_required
def api_history():
    rows = get_history(session["user_id"], request.args.get("search", ""))
    return jsonify({"success": True, "history": [dict(row) for row in rows]})


@app.route("/api/dashboard")
@login_required
def api_dashboard():
    user = current_user()
    payload = {"success": True, "user": dict(user)}
    if user["is_admin"]:
        payload["stats"] = get_user_stats()
    return jsonify(payload)


def respond(success, message, template_name=None, status=200, redirect_url=None):
    if wants_json() or request.path.startswith("/api"):
        payload = {"success": success, "message": message}
        if redirect_url:
            payload["redirect_url"] = redirect_url
        return jsonify(payload), status
    if redirect_url:
        return redirect(redirect_url)
    return render_template(template_name, message=message, success=success), status


@app.template_filter("datetime")
def format_datetime(value):
    if not value:
        return ""
    try:
        return datetime.fromisoformat(value).strftime("%d %b %Y, %I:%M %p")
    except ValueError:
        return value


with app.app_context():
    init_db()
    load_ml_artifacts()

app.teardown_appcontext(close_db)

@app.route("/debug")
def debug():
    import sklearn

    return {
        "sklearn_version": sklearn.__version__,
        "vectorizer_type": str(type(vectorizer)),
        "has_idf": hasattr(vectorizer, "idf_"),
        "idf_attr": "idf_" in dir(vectorizer),
        "vocab_size": len(vectorizer.vocabulary_) if hasattr(vectorizer, "vocabulary_") else 0,
    }


if __name__ == "__main__":
    app.run(debug=True)
