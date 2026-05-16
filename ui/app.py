import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, request, redirect, url_for, abort
import numpy as np
import sqlite3
import joblib
import tensorflow as tf
from flask_login import (
    LoginManager, UserMixin,
    login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import check_password_hash
from .database import init_db

app = Flask(__name__)
app.secret_key = "very-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DB_PATH = os.path.join(PROJECT_DIR, "patients.db")

init_db(DB_PATH)

login_manager = LoginManager(app)
login_manager.login_view = "login"


class User(UserMixin):
    def __init__(self, id, username, role):
        self.id = id
        self.username = username
        self.role = role


@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role FROM users WHERE id=?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return User(*row)
    return None


MODEL_DIR = os.path.join(PROJECT_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]
model_file = sorted(model_files)[-1] if model_files else None

if model_file:
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, model_file))
elif os.path.exists(os.path.join(PROJECT_DIR, "saved_model.keras")):
    model = tf.keras.models.load_model(os.path.join(PROJECT_DIR, "saved_model.keras"))
else:
    model = None

scaler_path = os.path.join(PROJECT_DIR, "scaler.pkl")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Required scaler file is missing: {scaler_path}")
scaler = joblib.load(scaler_path)


def role_required(role):
    def wrapper(fn):
        def decorated(*args, **kwargs):
            if current_user.role != role:
                abort(403)
            return fn(*args, **kwargs)
        decorated.__name__ = fn.__name__
        return decorated
    return wrapper


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, password, role FROM users WHERE username=?",
            (username,)
        )
        row = cursor.fetchone()
        conn.close()

        if row and check_password_hash(row[2], password):
            login_user(User(row[0], row[1], row[3]))
            return redirect(url_for("index"))

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
@login_required
@role_required("doctor")
def index():
    prediction = score = None

    if request.method == "POST" and model is not None:
        features = [float(request.form[x]) for x in [
            "pregnancies","glucose","bloodpressure",
            "skinthickness","insulin","bmi","dpf","age"
        ]]

        data = np.array(features).reshape(1, -1)
        data = scaler.transform(data)

        score = float(model.predict(data, verbose=0)[0][0])

        prediction = (
            "Diabetic" if score >= 0.7 else
            "Pre-Diabetic" if score >= 0.4 else
            "Non-Diabetic"
        )

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO patients (
            doctor, pregnancies, glucose, bloodpressure,
            skinthickness, insulin, bmi, dpf, age,
            prediction, probability
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (current_user.username, *features, prediction, score))
        conn.commit()
        conn.close()

    return render_template("index.html", prediction=prediction, score=score)


@app.route("/admin")
@login_required
@role_required("admin")
def admin():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients")
    records = cursor.fetchall()
    conn.close()
    return render_template("admin.html", records=records)


if __name__ == "__main__":
    app.run(debug=True)
