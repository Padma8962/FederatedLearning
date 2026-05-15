import sqlite3
from werkzeug.security import generate_password_hash


def init_db():
    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT
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
        probability REAL
    )
    """)

    users = [
        ("admin", generate_password_hash("admin123"), "admin"),
        ("doctor", generate_password_hash("doctor123"), "doctor")
    ]

    for user in users:
        cursor.execute("SELECT * FROM users WHERE username=?", (user[0],))
        if cursor.fetchone() is None:
            cursor.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                user
            )

    conn.commit()
    conn.close()