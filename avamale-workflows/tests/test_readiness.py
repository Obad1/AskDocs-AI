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
