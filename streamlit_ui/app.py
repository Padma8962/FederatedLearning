import os
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from werkzeug.security import check_password_hash, generate_password_hash

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ui.database import init_db


DB_PATH = Path(os.environ.get("DATABASE_PATH", PROJECT_DIR / "patients.db"))
SCALER_PATH = PROJECT_DIR / "scaler.pkl"

FEATURES = [
    ("pregnancies", "Pregnancies", 1.0, 1.0, "%.0f"),
    ("glucose", "Glucose", 120.0, 1.0, "%.1f"),
    ("bloodpressure", "Blood pressure", 70.0, 1.0, "%.1f"),
    ("skinthickness", "Skin thickness", 20.0, 1.0, "%.1f"),
    ("insulin", "Insulin", 80.0, 1.0, "%.1f"),
    ("bmi", "BMI", 25.0, 0.1, "%.1f"),
    ("dpf", "Diabetes pedigree function", 0.5, 0.01, "%.3f"),
    ("age", "Age", 30.0, 1.0, "%.0f"),
]

st.set_page_config(
    page_title="Federated Health",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    [data-testid="stMetric"] {
        background: rgba(28, 131, 225, 0.08);
        border: 1px solid rgba(28, 131, 225, 0.18);
        border-radius: 0.75rem;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def connect_db():
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def query_all(sql, parameters=()):
    with connect_db() as connection:
        return connection.execute(sql, parameters).fetchall()


def find_model_path():
    candidates = sorted((PROJECT_DIR / "saved_models").glob("*.keras"))
    if candidates:
        return candidates[-1]

    fallback = PROJECT_DIR / "saved_model.keras"
    return fallback if fallback.exists() else None


@st.cache_resource(show_spinner="Loading prediction model…")
def load_prediction_assets(model_path, scaler_path):
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler


def authenticate(username, password, role):
    with connect_db() as connection:
        row = connection.execute(
            """
            SELECT id, username, password, role, full_name, email
            FROM users
            WHERE username=?
            """,
            (username.strip(),),
        ).fetchone()

    if not row or row["role"] != role:
        return None
    if not check_password_hash(row["password"], password):
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "full_name": row["full_name"] or row["username"],
        "email": row["email"] or "",
    }


def register_user(full_name, email, username, password, role):
    if role not in {"doctor", "patient"}:
        raise ValueError("Invalid account role.")

    try:
        with connect_db() as connection:
            connection.execute(
                """
                INSERT INTO users (username, password, role, full_name, email)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    username.strip(),
                    generate_password_hash(password),
                    role,
                    full_name.strip(),
                    email.strip(),
                ),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError("That username is already registered.") from exc


def log_out():
    st.session_state.pop("user", None)
    st.rerun()


def show_login():
    st.title("🩺 Federated Health")
    st.caption("Secure diabetes-risk screening and patient follow-up")

    login_tab, signup_tab = st.tabs(["Sign in", "Create account"])

    with login_tab:
        with st.form("login_form"):
            role = st.segmented_control(
                "Account type",
                ["doctor", "patient", "admin"],
                default="doctor",
                format_func=str.title,
            )
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary")

        if submitted:
            user = authenticate(username, password, role)
            if user:
                st.session_state.user = user
                st.rerun()
            st.error("The username, password, or account type is incorrect.")

        with st.expander("Demo accounts"):
            st.code(
                "Doctor: doctor / doctor123\n"
                "Patient: patient / patient123\n"
                "Admin: admin / admin123"
            )

    with signup_tab:
        with st.form("signup_form", clear_on_submit=True):
            full_name = st.text_input("Full name")
            email = st.text_input("Email")
            new_username = st.text_input("Choose a username")
            new_password = st.text_input("Choose a password", type="password")
            new_role = st.selectbox(
                "Account type",
                ["patient", "doctor"],
                format_func=str.title,
            )
            registered = st.form_submit_button("Create account", type="primary")

        if registered:
            if not all(
                value.strip()
                for value in [full_name, email, new_username, new_password]
            ):
                st.error("Complete every field.")
            elif len(new_password) < 6:
                st.error("Use a password with at least six characters.")
            else:
                try:
                    register_user(
                        full_name,
                        email,
                        new_username,
                        new_password,
                        new_role,
                    )
                    user = authenticate(new_username, new_password, new_role)
                    if user is None:
                        raise ValueError(
                            "The account was created, but automatic sign-in failed."
                        )
                    st.session_state.user = user
                    st.session_state.account_created = True
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))


def records_table(records, include_patient=False, include_doctor=False):
    columns = []
    if include_doctor:
        columns.append(("Doctor", "doctor"))
    if include_patient:
        columns.append(("Patient", "patient_username"))
    columns.extend(
        [
            ("Prediction", "prediction"),
            ("Probability", "probability"),
            ("Notes", "notes"),
            ("Created", "created_at"),
        ]
    )

    data = []
    for record in records:
        item = {}
        for label, key in columns:
            value = record[key]
            if key == "probability" and value is not None:
                value = f"{value * 100:.1f}%"
            elif not value:
                value = "—"
            item[label] = value
        data.append(item)

    st.dataframe(data, use_container_width=True, hide_index=True)


def doctor_dashboard(user):
    st.title("Doctor dashboard")
    st.caption("Run a diabetes-risk prediction and share the result with a patient.")

    patients = query_all(
        """
        SELECT username, full_name
        FROM users
        WHERE role='patient'
        ORDER BY full_name, username
        """
    )
    patient_options = {
        "Unassigned": None,
        **{
            f"{row['full_name'] or row['username']} (@{row['username']})": row[
                "username"
            ]
            for row in patients
        },
    }

    model_path = find_model_path()
    if model_path is None or not SCALER_PATH.exists():
        st.error("The prediction model or scaler is missing from this deployment.")
        return

    with st.form("prediction_form"):
        selected_label = st.selectbox("Patient", list(patient_options))
        values = {}
        left, right = st.columns(2)
        for index, (key, label, default, step, number_format) in enumerate(FEATURES):
            target = left if index % 2 == 0 else right
            values[key] = target.number_input(
                label,
                min_value=0.0,
                value=default,
                step=step,
                format=number_format,
            )
        notes = st.text_area("Doctor notes", placeholder="Optional care note")
        submitted = st.form_submit_button("Run prediction", type="primary")

    if submitted:
        try:
            model, scaler = load_prediction_assets(str(model_path), str(SCALER_PATH))
            feature_values = [values[key] for key, *_ in FEATURES]
            scaled_data = scaler.transform(
                np.asarray(feature_values, dtype=float).reshape(1, -1)
            )
            score = float(model.predict(scaled_data, verbose=0)[0][0])
            prediction = (
                "Diabetic"
                if score >= 0.7
                else "Pre-Diabetic"
                if score >= 0.4
                else "Non-Diabetic"
            )

            with connect_db() as connection:
                connection.execute(
                    """
                    INSERT INTO patients (
                        doctor, pregnancies, glucose, bloodpressure,
                        skinthickness, insulin, bmi, dpf, age,
                        prediction, probability, patient_username, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user["username"],
                        *feature_values,
                        prediction,
                        score,
                        patient_options[selected_label],
                        notes.strip() or None,
                    ),
                )

            st.session_state.latest_result = (prediction, score)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    if "latest_result" in st.session_state:
        prediction, score = st.session_state.latest_result
        result_col, score_col = st.columns(2)
        result_col.metric("Result", prediction)
        score_col.metric("Model probability", f"{score * 100:.1f}%")

    st.subheader("Recent predictions")
    records = query_all(
        """
        SELECT *
        FROM patients
        WHERE doctor=?
        ORDER BY id DESC
        LIMIT 8
        """,
        (user["username"],),
    )
    if records:
        records_table(records, include_patient=True)
    else:
        st.info("No predictions have been recorded yet.")


def patient_dashboard(user):
    st.title("Patient dashboard")
    st.caption("Review your prediction history and your doctor's notes.")

    records = query_all(
        """
        SELECT *
        FROM patients
        WHERE patient_username=?
        ORDER BY id DESC
        """,
        (user["username"],),
    )

    if not records:
        st.info("No prediction has been assigned to your account yet.")
        return

    latest = records[0]
    result_col, score_col, doctor_col = st.columns(3)
    result_col.metric("Latest result", latest["prediction"])
    score_col.metric("Model probability", f"{latest['probability'] * 100:.1f}%")
    doctor_col.metric("Doctor", latest["doctor"] or "—")

    st.subheader("Latest note")
    st.write(latest["notes"] or "No doctor notes were added.")

    st.subheader("Prediction history")
    records_table(records, include_doctor=True)


def admin_dashboard():
    st.title("Administrator dashboard")
    st.caption("Review activity across the screening application.")

    records = query_all("SELECT * FROM patients ORDER BY id DESC")
    user_count = query_all("SELECT COUNT(*) AS count FROM users")[0]["count"]

    users_col, prediction_col = st.columns(2)
    users_col.metric("Registered users", user_count)
    prediction_col.metric("Predictions", len(records))

    st.subheader("All predictions")
    if records:
        records_table(records, include_patient=True, include_doctor=True)
    else:
        st.info("No predictions have been recorded yet.")


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    init_db(str(DB_PATH))

    user = st.session_state.get("user")
    if not user:
        show_login()
        return

    if st.session_state.pop("account_created", False):
        st.success("Your account was created successfully.")

    with st.sidebar:
        st.subheader(user["full_name"])
        st.caption(f"{user['role'].title()} · @{user['username']}")
        st.button("Sign out", on_click=log_out, use_container_width=True)
        st.divider()
        st.caption(
            "Predictions are decision-support outputs and do not replace a "
            "clinical diagnosis."
        )

    if user["role"] == "doctor":
        doctor_dashboard(user)
    elif user["role"] == "patient":
        patient_dashboard(user)
    elif user["role"] == "admin":
        admin_dashboard()
    else:
        st.error("This account has an unsupported role.")


if __name__ == "__main__":
    main()
