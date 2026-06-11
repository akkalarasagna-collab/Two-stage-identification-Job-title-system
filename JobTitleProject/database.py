import os
import sqlite3
from datetime import datetime

from flask import g


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "jobtitle.db")


def get_db():
    if "db" not in g:
        os.makedirs(DATABASE_DIR, exist_ok=True)
        g.db = sqlite3.connect(DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = get_db()
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            job_description TEXT NOT NULL,
            predicted_job_title TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        """
    )
    db.commit()
    ensure_default_admin(db)


def ensure_default_admin(db):
    from werkzeug.security import generate_password_hash

    admin = db.execute("SELECT id FROM users WHERE email = ?", ("admin@jobtitle.local",)).fetchone()
    if admin:
        return
    db.execute(
        """
        INSERT INTO users (username, email, password_hash, is_admin, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "admin",
            "admin@jobtitle.local",
            generate_password_hash("admin123"),
            1,
            datetime.utcnow().isoformat(),
        ),
    )
    db.commit()
