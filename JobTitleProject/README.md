# Two-Stage Job Title Identification System for Online Job Advertisements

This BTech mini project is a complete Flask web application that predicts the most suitable job title from a job description. It uses the already trained `job_title_classifier.joblib` and `tfidf_vectorizer.joblib` files directly inside Flask.

## Project Abstract

Online job advertisements often contain long, unstructured descriptions that make title identification difficult for job portals, recruiters and applicants. This project implements a two-stage job title identification system. In the first stage, the input job description is cleaned using NLP preprocessing. In the second stage, TF-IDF features are passed to a trained LinearSVC classifier to predict the most appropriate job title. The application also includes user authentication, prediction history, admin analytics and deployment-ready configuration.

## Problem Statement

To design and develop an end-to-end web application that accepts an online job advertisement description and predicts the correct job title using Natural Language Processing and Machine Learning. The system should store user predictions, provide searchable history and offer an admin dashboard for monitoring usage and prediction distribution.

## System Architecture

User Interface:
HTML, CSS and JavaScript pages for signup, login, prediction, history and admin analytics.

Backend:
Flask routes manage authentication, sessions, prediction requests, history and admin dashboards.

Machine Learning Layer:
NLTK preprocessing cleans text. The saved TF-IDF vectorizer converts text into numerical features. The saved LinearSVC model predicts the job title.

Database:
SQLite stores users and prediction history.

Deployment:
Gunicorn and `Procfile` support Render or Railway deployment.

## Flow Diagram

```text
User
  |
  v
Signup/Login
  |
  v
Dashboard
  |
  v
Enter Job Description
  |
  v
NLP Preprocessing
  |
  v
TF-IDF Vectorizer
  |
  v
LinearSVC Classifier
  |
  v
Predicted Job Title + Confidence Score
  |
  v
Save Prediction in SQLite
  |
  v
History / Admin Analytics
```

## Technology Justification

- Python is suitable for NLP and machine learning development.
- Flask is lightweight and ideal for mini projects and deployable ML applications.
- NLTK provides reliable text preprocessing utilities.
- TF-IDF is effective for converting job descriptions into meaningful numeric features.
- LinearSVC performs well for high-dimensional text classification.
- SQLite is simple, portable and sufficient for academic project storage.
- HTML, CSS and JavaScript provide a responsive browser-based interface.
- Gunicorn, Render and Railway make deployment straightforward.

## Project Structure

```text
JobTitleProject/
├── app.py
├── train.py
├── database.py
├── models.py
├── requirements.txt
├── Procfile
├── README.md
├── job_title_classifier.joblib
├── tfidf_vectorizer.joblib
├── job_title_des.csv
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── images/
│       ├── app-background.svg
│       └── hero-workspace.svg
├── templates/
│   ├── base.html
│   ├── welcome.html
│   ├── login.html
│   ├── signup.html
│   ├── dashboard.html
│   ├── history.html
│   └── admin_dashboard.html
└── database/
    └── jobtitle.db
```

## Required Model Files

Place these existing trained files in the `JobTitleProject/` root:

```text
job_title_classifier.joblib
tfidf_vectorizer.joblib
```

The application does not use dummy ML code. It loads these files with `joblib.load()` in `app.py`.

## Local Setup

```bash
cd JobTitleProject
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Default admin login:

```text
Email: admin@jobtitle.local
Password: admin123
```

Change the admin password before public deployment.

## REST API Endpoints

```text
POST /signup
POST /login
GET  /logout
POST /predict
GET  /history
GET  /dashboard
```

JSON-friendly API aliases:

```text
POST /api/signup
POST /api/login
GET  /api/logout
GET  /api/history
GET  /api/dashboard
```

Prediction request example:

```json
{
  "job_description": "We are looking for a Python developer with Flask, SQL and machine learning experience."
}
```

## Database Tables

users:

```text
id
username
email
password_hash
is_admin
created_at
```

predictions:

```text
id
user_id
job_description
predicted_job_title
timestamp
```

## Render Deployment

1. Push the project to GitHub.
2. Create a new Web Service on Render.
3. Select the GitHub repository.
4. Set build command:

```bash
pip install -r requirements.txt
```

5. Set start command:

```bash
gunicorn app:app
```

6. Add environment variable:

```text
SECRET_KEY=your-secure-secret-key
```

7. Ensure the trained `.joblib` files are included in the deployed project or uploaded through your deployment workflow.

## Railway Deployment

1. Push the project to GitHub.
2. Create a Railway project from the repository.
3. Railway will detect Python dependencies from `requirements.txt`.
4. Add environment variable:

```text
SECRET_KEY=your-secure-secret-key
```

5. The `Procfile` starts the app using Gunicorn.

## Future Scope

- Add probability calibration using `CalibratedClassifierCV`.
- Add user profile management and password reset.
- Add CSV upload for bulk job title prediction.
- Add model performance monitoring for admin users.
- Move from SQLite to PostgreSQL for production scale.
- Add role-based access controls for multiple admin types.
- Improve preprocessing with spaCy or transformer embeddings.

## References

- Flask documentation: https://flask.palletsprojects.com/
- Scikit-learn documentation: https://scikit-learn.org/
- NLTK documentation: https://www.nltk.org/
- SQLite documentation: https://sqlite.org/docs.html
- Render documentation: https://render.com/docs
- Railway documentation: https://docs.railway.com/
