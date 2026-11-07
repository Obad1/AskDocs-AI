"""Database readiness is independent of process liveness and does not leak errors."""
from fastapi.testclient import TestClient

from product_app.api import create_app


def test_readiness_reports_database_available(tmp_path, monkeypatch):
    monkeypatch.setenv('AVAMALE_STORAGE_BACKEND', 'sqlite')
    monkeypatch.delenv('AVAMALE_DATABASE_URL', raising=False)
    app = create_app(
        db_path=str(tmp_path / 'ready.sqlite3'),
        admin_password='synthetic-password-at-least-twelve',
        session_secret='synthetic-session-secret-at-least-32-characters',
    )
    with TestClient(app) as client:
        assert client.get('/healthz').status_code == 200
        response = client.get('/readyz')
        assert response.status_code == 200
        assert response.json() == {'ok': True, 'database': 'available'}


def test_readiness_fails_closed_but_liveness_stays_healthy(tmp_path, monkeypatch):
    monkeypatch.setenv('AVAMALE_STORAGE_BACKEND', 'sqlite')
    monkeypatch.delenv('AVAMALE_DATABASE_URL', raising=False)
    app = create_app(
        db_path=str(tmp_path / 'ready.sqlite3'),
        admin_password='synthetic-password-at-least-twelve',
        session_secret='synthetic-session-secret-at-least-32-characters',
    )

    def offline():
        raise RuntimeError('secret database connection error must not leak')

    monkeypatch.setattr(app.state.store, 'conn', offline)
    with TestClient(app) as client:
        response = client.get('/readyz')
        assert response.status_code == 503
        assert response.json() == {'ok': False, 'database': 'unavailable'}
        assert client.get('/healthz').status_code == 200


def test_readiness_recovers_after_transient_connection_failure(tmp_path, monkeypatch):
    """A healthy connection after a temporary outage restores readiness automatically."""
    monkeypatch.setenv('AVAMALE_STORAGE_BACKEND', 'sqlite')
    monkeypatch.delenv('AVAMALE_DATABASE_URL', raising=False)
    app = create_app(
        db_path=str(tmp_path / 'recover.sqlite3'),
        admin_password='synthetic-password-at-least-twelve',
        session_secret='synthetic-session-secret-at-least-32-characters',
    )
    original = app.state.store.conn
    attempts = 0

    def transient_failure():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError('transient database outage')
        return original()

    monkeypatch.setattr(app.state.store, 'conn', transient_failure)
    with TestClient(app) as client:
        first = client.get('/readyz')
        assert first.status_code == 503
        assert 'transient database outage' not in first.text
        second = client.get('/readyz')
        assert second.status_code == 200
        assert second.json() == {'ok': True, 'database': 'available'}
        assert client.get('/healthz').status_code == 200
