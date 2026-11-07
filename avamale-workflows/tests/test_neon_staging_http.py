"""Optional real PostgreSQL HTTP acceptance test (synthetic only).

Requires a dedicated, disposable database whose ID has been explicitly confirmed.
Never point this test at a customer, shared, or production database.
"""
import os
from io import BytesIO
from pathlib import Path

import pytest

RUN_LIVE = os.getenv('AVAMALE_RUN_NEON_ACCEPTANCE') == '1'
DATABASE_URL = os.getenv('AVAMALE_TEST_DATABASE_URL', '')
SAFE_NAME = os.getenv('AVAMALE_TEST_DATABASE_NAME', '')

pytestmark = pytest.mark.skipif(
    not (RUN_LIVE and DATABASE_URL and SAFE_NAME == 'avamale_synthetic'),
    reason='Guarded live acceptance requires explicit synthetic Neon test configuration',
)


def test_live_postgresql_all_eighteen_http_and_recreated_app(monkeypatch):
    from fastapi.testclient import TestClient
    from openpyxl import load_workbook
    from product_app.api import create_app
    from product_app.catalog import PRODUCTS

    assert '/avamale_synthetic' in DATABASE_URL.split('?')[0]
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv('AVAMALE_STORAGE_BACKEND', 'postgres')
    monkeypatch.setenv('AVAMALE_DATABASE_URL', DATABASE_URL)
    monkeypatch.delenv('AVAMALE_DB_PATH', raising=False)
    monkeypatch.setenv('AVAMALE_PUBLIC', '0')
    monkeypatch.setenv('AVAMALE_COOKIE_SECURE', '0')
    monkeypatch.setenv('AVAMALE_ALLOW_LEADS', '0')

    password = 'synthetic-only-test-password-2026'
    secret = 'synthetic-only-session-secret-longer-than-32-characters'
    created = []

    with TestClient(create_app(admin_password=password, session_secret=secret)) as client:
        assert client.get('/readyz').json() == {'ok': True, 'database': 'available'}
        assert len(client.get('/api/products').json()) == 18
        assert client.get('/api/runs').status_code == 401
        login = client.post('/api/auth/login', json={'password': password})
        assert login.status_code == 200
        headers = {'X-CSRF-Token': login.json()['csrf']}

        for code in sorted(PRODUCTS):
            if code == 'AA02':
                files = {
                    'bank': ('bank_synthetic.csv', (root/'samples/bank_synthetic.csv').read_bytes(), 'text/csv'),
                    'ledger': ('ledger_synthetic.csv', (root/'samples/ledger_synthetic.csv').read_bytes(), 'text/csv'),
                }
            else:
                filename = 'freight_synthetic.csv' if code == 'AA03' else f'{code}.csv'
                files = {'data': (filename, (root/'samples'/filename).read_bytes(), 'text/csv')}
            response = client.post('/api/runs/'+code,
                                   data={'synthetic':'true','as_of':'2026-11-07','date_tolerance':'1'},
                                   files=files, headers=headers)
            assert response.status_code == 200, (code, response.text)
            rid = response.json()['id']
            created.append((code, rid))
            update = client.post(f'/api/runs/{rid}/findings/0',
                                 json={'status':'accepted','note':'Invented example only','reviewer':'test-operator'},
                                 headers=headers)
            assert update.status_code == 200, (code, update.text)
            assert [x['action'] for x in client.get(f'/api/runs/{rid}/history').json()] == ['created', 'decision']
            assert client.get(f'/api/runs/{rid}/export.csv').status_code == 200
            assert client.get(f'/api/runs/{rid}/export.json').status_code == 200
            xlsx = client.get(f'/api/runs/{rid}/export.xlsx')
            assert xlsx.status_code == 200
            book = load_workbook(BytesIO(xlsx.content), read_only=True)
            assert book['Read me']['B1'].value == 'SYNTHETIC-ONLY / DEMONSTRATION'

    # A separate FastAPI instance with newly created database connections.
    with TestClient(create_app(admin_password=password, session_secret=secret)) as client:
        assert client.post('/api/auth/login', json={'password': password}).status_code == 200
        for code, rid in created:
            response = client.get(f'/api/runs/{rid}')
            assert response.status_code == 200, (code, response.text)
            assert response.json()['product'] == code
            assert response.json()['decisions'][0]['state'] == 'accepted'
            assert [x['action'] for x in client.get(f'/api/runs/{rid}/history').json()] == ['created', 'decision']


def test_postgresql_concurrent_review_preserves_both_decisions(monkeypatch):
    """Two reviewers updating separate findings should not lose audit entries."""
    from concurrent.futures import ThreadPoolExecutor
    from product_app.postgres_store import PostgresStore

    assert '/avamale_synthetic' in DATABASE_URL.split('?')[0]
    store = PostgresStore(DATABASE_URL)
    run_id = store.create_run(
        'AA01', 'Synthetic concurrent reviewers', {}, {},
        [{'status': 'REVIEW_REQUIRED', 'review_required': True},
         {'status': 'REVIEW_REQUIRED', 'review_required': True}],
    )

    def decide(index):
        return store.update_decision(
            run_id, index, 'accepted', f'Synthetic review {index}', f'reviewer-{index}',
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(decide, i) for i in (0, 1)]
        results = [future.result(timeout=20) for future in futures]

    assert all(item['state'] in ('in_review', 'review_complete') for item in results)
    final = store.get_run(run_id)
    assert final['state'] == 'review_complete'
    assert [item['state'] for item in final['decisions']] == ['accepted', 'accepted']
    events = store.events(run_id)
    assert [item['action'] for item in events] == ['created', 'decision', 'decision']
    assert {item['actor'] for item in events[1:]} == {'reviewer-0', 'reviewer-1'}


def test_postgresql_failed_review_preserves_record_and_audit(monkeypatch):
    """Invalid requests must leave both review decisions and audit history untouched."""
    from product_app.postgres_store import PostgresStore
    from product_app.store import StorageError

    assert '/avamale_synthetic' in DATABASE_URL.split('?')[0]
    store = PostgresStore(DATABASE_URL)
    run_id = store.create_run(
        'AA01', 'Synthetic rejection guard', {}, {},
        [{'status': 'REVIEW_REQUIRED', 'review_required': True}],
    )
    before = store.get_run(run_id)
    history_before = store.events(run_id)
    with pytest.raises(StorageError, match='Finding index'):
        store.update_decision(run_id, 5, 'accepted', 'Invalid index', 'synthetic-operator')
    with pytest.raises(StorageError, match='Unsupported review status'):
        store.update_decision(run_id, 0, 'invalid', 'Invalid state', 'synthetic-operator')
    assert store.get_run(run_id) == before
    assert store.events(run_id) == history_before
