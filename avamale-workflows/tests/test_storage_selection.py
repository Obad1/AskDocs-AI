"""Storage backend selection must not affect other apps sharing an environment."""
import pytest
from product_app.api import create_app

PASSWORD = 'synthetic-test-password-2026'
SECRET = 'synthetic-test-cookie-signing-key-long-enough-2026'


def test_sqlite_default_preserved(tmp_path, monkeypatch):
    monkeypatch.delenv('AVAMALE_DATABASE_URL', raising=False)
    monkeypatch.delenv('AVAMALE_DB_PATH', raising=False)
    monkeypatch.delenv('AVAMALE_STORAGE_BACKEND', raising=False)
    app = create_app(db_path=str(tmp_path / 'local.sqlite3'),
                     admin_password=PASSWORD, session_secret=SECRET)
    assert app.state.store.__class__.__name__ == 'Store'


def test_invalid_postgres_url_fails_closed(monkeypatch):
    monkeypatch.setenv('AVAMALE_STORAGE_BACKEND', 'postgres')
    monkeypatch.delenv('AVAMALE_DB_PATH', raising=False)
    monkeypatch.delenv('AVAMALE_DATABASE_URL', raising=False)
    with pytest.raises(RuntimeError, match='AVAMALE_DATABASE_URL'):
        create_app(admin_password=PASSWORD, session_secret=SECRET)


def test_postgres_rejects_sqlite_path(monkeypatch, tmp_path):
    monkeypatch.setenv('AVAMALE_STORAGE_BACKEND', 'postgres')
    monkeypatch.setenv('AVAMALE_DATABASE_URL', 'postgresql://placeholder.invalid/test')
    monkeypatch.delenv('AVAMALE_DB_PATH', raising=False)
    with pytest.raises(RuntimeError, match='SQLite'):
        create_app(db_path=str(tmp_path / 'not-used.sqlite3'),
                   admin_password=PASSWORD, session_secret=SECRET)


def test_sqlite_ignores_shared_generic_database_url(monkeypatch, tmp_path):
    monkeypatch.setenv('AVAMALE_STORAGE_BACKEND', 'sqlite')
    monkeypatch.setenv('DATABASE_URL', 'postgresql://another-app.invalid/shared')
    monkeypatch.delenv('AVAMALE_DATABASE_URL', raising=False)
    app = create_app(db_path=str(tmp_path / 'local.sqlite3'),
                     admin_password=PASSWORD, session_secret=SECRET)
    assert app.state.store.__class__.__name__ == 'Store'


def test_unknown_backend_fails_closed(monkeypatch):
    monkeypatch.setenv('AVAMALE_STORAGE_BACKEND', 'unsupported')
    monkeypatch.delenv('AVAMALE_DATABASE_URL', raising=False)
    with pytest.raises(RuntimeError, match='Unknown AVAMALE_STORAGE_BACKEND'):
        create_app(admin_password=PASSWORD, session_secret=SECRET)
