"""Storage backend configuration checks for isolated Neon migration.

Does not connect to Neon or modify any infrastructure.
"""
import pytest
from product_app.api import create_app

PASSWORD = 'synthetic-test-password-2026'
SECRET = 'synthetic-test-cookie-signing-key-long-enough-2026'


def test_sqlite_default_preserved(tmp_path, monkeypatch):
    monkeypatch.delenv('DATABASE_URL', raising=False)
    monkeypatch.delenv('AVAMALE_DB_PATH', raising=False)
    monkeypatch.delenv('AVAMALE_STORAGE_BACKEND', raising=False)
    app = create_app(db_path=str(tmp_path / 'local.sqlite3'),
                     admin_password=PASSWORD, session_secret=SECRET)
    assert len(app.routes) > 0
    assert app.state.store.__class__.__name__ == 'Store'


def test_invalid_postgres_url_fails_closed(monkeypatch):
    monkeypatch.setenv('AVAMALE_STORAGE_BACKEND', 'postgres')
    monkeypatch.delenv('AVAMALE_DB_PATH', raising=False)
    monkeypatch.delenv('DATABASE_URL', raising=False)
    with pytest.raises(RuntimeError, match='DATABASE_URL'):
        create_app(admin_password=PASSWORD, session_secret=SECRET)


def test_postgres_rejects_sqlite_path(monkeypatch, tmp_path):
    monkeypatch.setenv('AVAMALE_STORAGE_BACKEND', 'postgres')
    monkeypatch.setenv('DATABASE_URL', 'postgresql://placeholder.invalid/test')
    monkeypatch.delenv('AVAMALE_DB_PATH', raising=False)
    with pytest.raises(RuntimeError, match='SQLite'):
        create_app(db_path=str(tmp_path / 'not-used.sqlite3'),
                   admin_password=PASSWORD, session_secret=SECRET)


def test_sqlite_rejects_accidental_neon_url(monkeypatch, tmp_path):
    monkeypatch.setenv('AVAMALE_STORAGE_BACKEND', 'sqlite')
    monkeypatch.setenv('DATABASE_URL', 'postgresql://placeholder.invalid/test')
    with pytest.raises(RuntimeError, match='DATABASE_URL'):
        create_app(db_path=str(tmp_path / 'not-used.sqlite3'),
                   admin_password=PASSWORD, session_secret=SECRET)


def test_unknown_backend_fails_closed(monkeypatch):
    monkeypatch.setenv('AVAMALE_STORAGE_BACKEND', 'unsupported')
    monkeypatch.delenv('DATABASE_URL', raising=False)
    with pytest.raises(RuntimeError, match='Unknown AVAMALE_STORAGE_BACKEND'):
        create_app(admin_password=PASSWORD, session_secret=SECRET)
