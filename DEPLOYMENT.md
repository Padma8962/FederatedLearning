# Deployment

This project has two separate processes:

- `web`: runs either the Streamlit or Flask healthcare UI.
- `worker`: runs the Flower federated-learning server.

## Streamlit Community Cloud

The Streamlit app includes role-based login, account creation, diabetes-risk
prediction, patient history, and the administrator view.

1. Push the repository to GitHub.
2. In Streamlit Community Cloud, create an app from the repository.
3. Set the main file path to `streamlit_ui/app.py`.
4. Deploy the app.

The `streamlit_ui/requirements.txt` file contains the Streamlit and prediction
dependencies without adding the large TensorFlow package to the Vercel
function. The app loads `saved_model.keras` and `scaler.pkl` from the project
root.

You can run it locally with:

```bash
streamlit run streamlit_ui/app.py
```

The default database is `patients.db`. Files written by Streamlit Community
Cloud are not durable across app restarts, so use the default database only
for demonstrations. Before storing real patient data, replace SQLite with a
managed database and add the appropriate security, consent, auditing, and
regulatory controls.

Streamlit hosts only the web application. Run the Flower server separately
using the worker instructions below; hospital clients cannot connect to a
Flower TCP server through the Streamlit app.

## Vercel

Vercel can host the Flask web UI as a Python Function. The project includes:

- `api/index.py`: imports the Flask app for Vercel.
- `vercel.json`: routes every request to the Flask function.
- `.vercelignore`: keeps local-only files out of the deployment bundle.

Deploy from the project root:

```bash
vercel
vercel --prod
```

Or import the GitHub repository in the Vercel dashboard.

Set this environment variable in Vercel:

```bash
SECRET_KEY=replace-with-a-long-random-value
```

Vercel deployments use a temporary SQLite database at `/tmp/patients.db`.
That is enough for demo logins and testing, but it is not persistent storage.
For production patient records, connect the app to a hosted database.

The Vercel build uses `requirements.txt`, which is intentionally lightweight.
The TensorFlow prediction model and Flower worker dependencies are kept in
`requirements-full.txt` because they are better suited to a persistent backend
host such as Render.

## Render / persistent backend

The public website link comes from the `web` process:

```bash
gunicorn ui.app:app
```

The federated backend must be deployed as a separate worker/service:

```bash
python server/server.py
```

On Render, use these start commands:

- Website web service: `gunicorn ui.app:app`
- Flower backend service: `python server/server.py`

Do not use `python server/server.py` for the public Flask website service.
On Render free instances, use `gunicorn --timeout 120 ui.app:app` if the
first prediction is slow after the service wakes up.

## Environment variables

For the Flask web app:

```bash
SECRET_KEY=replace-with-a-long-random-value
DATABASE_PATH=patients.db
```

For the Flower server:

```bash
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
```

Some platforms provide `PORT` automatically. The server will use `SERVER_PORT`
first, then `PORT`, then `8080`.

For hospital clients connecting to a deployed Flower server:

```bash
FLOWER_SERVER_ADDRESS=your-backend-host:8080
```

## Login

Default demo accounts are created automatically when the web app starts:

- Doctor: `doctor` / `doctor123`
- Patient: `patient` / `patient123`
- Admin: `admin` / `admin123`

## Notes

The Flask website and Flower server are not the same thing. If only the `web`
process is deployed, the website can open, but the federated-learning backend
will not be running.
