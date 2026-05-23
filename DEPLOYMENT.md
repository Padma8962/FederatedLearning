# Deployment

This project has two separate processes:

- `web`: runs the Flask healthcare UI.
- `worker`: runs the Flower federated-learning server.

The public website link comes from the `web` process:

```bash
gunicorn ui.app:app
```

The federated backend must be deployed as a separate worker/service:

```bash
python server/server.py
```

## Environment variables

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
- Admin: `admin` / `admin123`

## Notes

The Flask website and Flower server are not the same thing. If only the `web`
process is deployed, the website can open, but the federated-learning backend
will not be running.
