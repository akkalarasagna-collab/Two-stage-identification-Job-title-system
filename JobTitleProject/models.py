from datetime import datetime

from database import get_db


def create_user(username, email, password_hash):
    db = get_db()
    db.execute(
        """
        INSERT INTO users (username, email, password_hash, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (username, email, password_hash, datetime.utcnow().isoformat()),
    )
    db.commit()


def get_user_by_email(email):
    return get_db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()


def get_user_by_username(username):
    return get_db().execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()


def get_user_by_id(user_id):
    return get_db().execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def create_prediction(user_id, job_description, predicted_job_title):
    db = get_db()
    db.execute(
        """
        INSERT INTO predictions (user_id, job_description, predicted_job_title, timestamp)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, job_description, predicted_job_title, datetime.utcnow().isoformat()),
    )
    db.commit()


def get_history(user_id, search=""):
    query = """
        SELECT * FROM predictions
        WHERE user_id = ?
    """
    params = [user_id]
    if search:
        query += " AND (job_description LIKE ? OR predicted_job_title LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])
    query += " ORDER BY timestamp DESC"
    return get_db().execute(query, params).fetchall()


def get_user_stats():
    db = get_db()
    total_users = db.execute("SELECT COUNT(*) AS count FROM users WHERE is_admin = 0").fetchone()["count"]
    total_predictions = db.execute("SELECT COUNT(*) AS count FROM predictions").fetchone()["count"]
    most_predicted = db.execute(
        """
        SELECT predicted_job_title, COUNT(*) AS count
        FROM predictions
        GROUP BY predicted_job_title
        ORDER BY count DESC
        LIMIT 8
        """
    ).fetchall()
    recent_activities = db.execute(
        """
        SELECT p.predicted_job_title, p.timestamp, u.username
        FROM predictions p
        JOIN users u ON u.id = p.user_id
        ORDER BY p.timestamp DESC
        LIMIT 10
        """
    ).fetchall()
    user_activity = db.execute(
        """
        SELECT u.username, COUNT(p.id) AS count
        FROM users u
        LEFT JOIN predictions p ON p.user_id = u.id
        WHERE u.is_admin = 0
        GROUP BY u.id
        ORDER BY count DESC
        LIMIT 8
        """
    ).fetchall()
    return {
        "total_users": total_users,
        "total_predictions": total_predictions,
        "most_predicted": [dict(row) for row in most_predicted],
        "recent_activities": [dict(row) for row in recent_activities],
        "user_activity": [dict(row) for row in user_activity],
    }
