# Usage telemetry (MySQL + FastAPI)

Telemetry is optional. The desktop app keeps projects and datasets in its existing local SQLite database. It sends only `installation_id`, `event_name`, `feature_name`, `status`, `duration_ms`, `app_version`, and `created_at` to the configured API. The sender constructs this fixed schema; it never reads or serializes dataset contents, file paths, analysis results, user input, or API keys. An opaque random installation ID is stored in the app's local data directory. No telemetry queue is written to SQLite.

## Local development

1. Start MySQL locally and open `mysql_setup.sql` in MySQL Workbench. Uncomment the account provisioning statements and replace the two secret placeholders with different random secrets. Keep the API account host restricted to the backend machine; for remote development, change `localhost` to that machine's specific IP, not `%`.
2. From the project root, create a backend virtual environment and install `backend/requirements.txt`. Copy `backend/.env.example` to `backend/.env` and set the API-account password. The API loads this file automatically for local development; environment variables already set by the process take precedence. Do not commit `.env` files.
3. Start/restart the service from the project root: `uvicorn backend.api:app --host 127.0.0.1 --port 8000`. The API's `/health` endpoint checks process availability; `POST /api/events` validates the event and stores it in MySQL.
4. Set `TELEMETRY_API_URL=http://127.0.0.1:8000` in the desktop app's `.env`, then launch the app. Leave it blank to disable telemetry. Desktop telemetry is best-effort and asynchronous.
5. In Workbench, run `SELECT * FROM vw_daily_usage ORDER BY report_date DESC;` and the total-summary query at the bottom of `mysql_setup.sql`.

The API schema rejects unknown fields and permits only the four required event names and defined status values. The event status must match its event name. For managed installations, set `TELEMETRY_API_TOKEN` on both ends: the desktop sends its SHA-256 digest in a Bearer header and the server checks it in constant time. Because a value shipped in a desktop app can be extracted, this is an abuse-control layer rather than strong client identity. Before production, put the API behind HTTPS, add rate limiting and abuse protection, restrict MySQL network access to the API host, use a unique secret-managed database password, and enable MySQL TLS certificate verification (`MYSQL_SSL_CA`).

For a production deployment, configure `TELEMETRY_API_URL` to the HTTPS API URL and configure MySQL credentials only in the backend service environment/secret manager. The API account has INSERT only; the admin account is separate and read-only. The desktop never connects directly to MySQL. Back up and retention-limit `app_events` according to the deployment's privacy policy.
