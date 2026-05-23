import sqlite3
from werkzeug.security import generate_password_hash


def add_column_if_missing(cursor, table, column, definition):
    cursor.execute(f"PRAGMA table_info({table})")
    existing_columns = {row[1] for row in cursor.fetchall()}
    if column not in existing_columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db(db_path="patients.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT,
        full_name TEXT,
        email TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doctor TEXT,
        pregnancies REAL,
        glucose REAL,
        bloodpressure REAL,
        skinthickness REAL,
        insulin REAL,
        bmi REAL,
        dpf REAL,
        age REAL,
        prediction TEXT,
        probability REAL,
        patient_username TEXT,
        notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    add_column_if_missing(cursor, "users", "full_name", "TEXT")
    add_column_if_missing(cursor, "users", "email", "TEXT")
    add_column_if_missing(cursor, "patients", "patient_username", "TEXT")
    add_column_if_missing(cursor, "patients", "notes", "TEXT")
    add_column_if_missing(cursor, "patients", "created_at", "TEXT")

    users = [
        ("admin", generate_password_hash("admin123"), "admin", "Admin User", "admin@example.com"),
        ("doctor", generate_password_hash("doctor123"), "doctor", "Demo Doctor", "doctor@example.com"),
        ("patient", generate_password_hash("patient123"), "patient", "Demo Patient", "patient@example.com"),
    ]

    for user in users:
        cursor.execute("SELECT * FROM users WHERE username=?", (user[0],))
        if cursor.fetchone() is None:
            cursor.execute(
                """
                INSERT INTO users (username, password, role, full_name, email)
                VALUES (?, ?, ?, ?, ?)
                """,
                user
            )

    conn.commit()
    conn.close()
