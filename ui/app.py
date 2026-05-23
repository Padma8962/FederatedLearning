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
from werkzeug.security import check_password_hash, generate_password_hash

try:
    from .database import init_db
except ImportError:
    from database import init_db

app = Flask(__name__)
app.secret_key = "very-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DB_PATH = os.path.join(PROJECT_DIR, "patients.db")

init_db(DB_PATH)

login_manager = LoginManager(app)
login_manager.login_view = "login"


class User(UserMixin):
    def __init__(self, id, username, role, full_name=None, email=None):
        self.id = id
        self.username = username
        self.role = role
        self.full_name = full_name or username
        self.email = email or ""


@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, role, full_name, email FROM users WHERE id=?",
        (user_id,),
    )
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
                return redirect_for_role(current_user.role)
            return fn(*args, **kwargs)
        decorated.__name__ = fn.__name__
        return decorated
    return wrapper


def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def redirect_for_role(role):
    if role == "doctor":
        return redirect(url_for("doctor_dashboard"))
    if role == "patient":
        return redirect(url_for("patient_dashboard"))
    if role == "admin":
        return redirect(url_for("admin"))
    return redirect(url_for("login"))


def authenticate(username, password, expected_role=None):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, username, password, role, full_name, email
        FROM users
        WHERE username=?
        """,
        (username,),
    )
    row = cursor.fetchone()
    conn.close()

    if not row or not check_password_hash(row["password"], password):
        return None
    if expected_role and row["role"] != expected_role:
        return None
    return User(row["id"], row["username"], row["role"], row["full_name"], row["email"])


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect_for_role(current_user.role)

    if request.method == "POST":
        role = request.form.get("role", "doctor")
        return redirect(url_for("role_login", role=role))

    return render_template("login.html")


@app.route("/login/<role>", methods=["GET", "POST"])
def role_login(role):
    if role not in ("doctor", "patient", "admin"):
        abort(404)

    if current_user.is_authenticated:
        return redirect_for_role(current_user.role)

    if request.method == "POST":
        user = authenticate(
            request.form["username"],
            request.form["password"],
            expected_role=role,
        )

        if user:
            login_user(user)
            return redirect_for_role(user.role)

        return render_template("login.html", selected_role=role, error="Invalid credentials for this role")

    return render_template("login.html", selected_role=role)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect_for_role(current_user.role)

    if request.method == "POST":
        username = request.form["username"].strip()
        full_name = request.form["full_name"].strip()
        email = request.form["email"].strip()
        password = request.form["password"]
        role = request.form.get("role", "patient")

        if role not in ("patient", "doctor"):
            role = "patient"

        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username=?", (username,))
        if cursor.fetchone():
            conn.close()
            return render_template("signup.html", error="Username already exists")

        cursor.execute(
            """
            INSERT INTO users (username, password, role, full_name, email)
            VALUES (?, ?, ?, ?, ?)
            """,
            (username, generate_password_hash(password), role, full_name, email),
        )
        conn.commit()
        conn.close()

        user = authenticate(username, password, expected_role=role)
        login_user(user)
        return redirect_for_role(role)

    return render_template("signup.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
@login_required
@role_required("doctor")
def doctor_dashboard():
    prediction = score = None
    selected_patient = notes = None

    if request.method == "POST" and model is not None:
        selected_patient = request.form.get("patient_username", "").strip() or None
        notes = request.form.get("notes", "").strip() or None
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
            prediction, probability, patient_username, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (current_user.username, *features, prediction, score, selected_patient, notes))
        conn.commit()
        conn.close()

    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT username, full_name FROM users WHERE role='patient' ORDER BY full_name, username"
    )
    patients = cursor.fetchall()
    cursor.execute(
        """
        SELECT *
        FROM patients
        WHERE doctor=?
        ORDER BY id DESC
        LIMIT 8
        """,
        (current_user.username,),
    )
    recent_records = cursor.fetchall()
    conn.close()

    return render_template(
        "index.html",
        prediction=prediction,
        score=score,
        patients=patients,
        recent_records=recent_records,
        selected_patient=selected_patient,
        model_available=model is not None,
    )


@app.route("/patient")
@login_required
@role_required("patient")
def patient_dashboard():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT *
        FROM patients
        WHERE patient_username=?
        ORDER BY id DESC
        """,
        (current_user.username,),
    )
    records = cursor.fetchall()
    conn.close()
    latest = records[0] if records else None
    return render_template("patient.html", records=records, latest=latest)


@app.route("/admin")
@login_required
@role_required("admin")
def admin():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients ORDER BY id DESC")
    records = cursor.fetchall()
    conn.close()
    return render_template("admin.html", records=records)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
