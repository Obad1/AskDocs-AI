"""AVAMALE Operations Studio: one runnable local pilot console for AA01-AA18.

Use synthetic data only. Do not deploy publicly without production security review,
independent regulatory sign-off for sensitive workflows, and privacy agreements.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import csv
import hmac
import hashlib
from io import BytesIO, StringIO
import os
from pathlib import Path
import re
import secrets
import time
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from avamale_review import InputError, safe_cell
from .catalog import PRODUCTS, catalog
from .engine import MAX_UPLOAD, run_product
from .store import Store, StorageError

APP_DIR = Path(__file__).resolve().parent
SESSION_COOKIE = 'avamale_demo_session'
SESSION_LIFETIME = 8 * 60 * 60


class LoginPayload(BaseModel):
    password: str


class DecisionPayload(BaseModel):
    status: str
    note: str = Field(default='', max_length=2000)
    reviewer: str = Field(default='operator', max_length=80)


class EnquiryPayload(BaseModel):
    product: str
    organization: str = Field(min_length=2, max_length=120)
    contact_email: str = Field(min_length=5, max_length=150)
    problem: str = Field(min_length=12, max_length=1400)
    consent: bool
    website: str = ''  # honeypot: never rely on it for spam protection alone


class EnquiryStatusPayload(BaseModel):
    status: str


def as_csv_cell(value):
    return safe_cell('' if value is None else str(value))


def create_app(*, db_path: str | None = None, admin_password: str | None = None,
               session_secret: str | None = None, allow_leads: bool | None = None) -> FastAPI:
    password = admin_password or os.getenv('AVAMALE_ADMIN_PASSWORD', '')
    key = session_secret or os.getenv('AVAMALE_SESSION_SECRET', '')
    if len(password) < 12 or len(key) < 32:
        raise RuntimeError('Set AVAMALE_ADMIN_PASSWORD (12+ chars) and AVAMALE_SESSION_SECRET (32+ chars)')
    public = os.getenv('AVAMALE_PUBLIC', '') == '1'
    secure_cookie = os.getenv('AVAMALE_COOKIE_SECURE', '') == '1'
    if public and not secure_cookie:
        raise RuntimeError('Public deployments must set AVAMALE_COOKIE_SECURE=1 behind HTTPS')
    backend = os.getenv('AVAMALE_STORAGE_BACKEND', 'sqlite').strip().lower()
    if backend == 'postgres':
        if db_path is not None or os.getenv('AVAMALE_DB_PATH'):
            raise RuntimeError('SQLite db_path/AVAMALE_DB_PATH cannot be used with PostgreSQL')
        database_url = os.getenv('AVAMALE_DATABASE_URL', '')
        if not database_url.startswith(('postgresql://', 'postgres://')):
            raise RuntimeError('PostgreSQL backend requires a valid AVAMALE_DATABASE_URL')
        from .postgres_store import PostgresStore
        database = PostgresStore(database_url)
    elif backend == 'sqlite':
        database = Store(db_path or os.getenv('AVAMALE_DB_PATH', './private_data/operations.sqlite3'))
    else:
        raise RuntimeError('Unknown AVAMALE_STORAGE_BACKEND; select sqlite or postgres')
    allow_enquiries = bool(allow_leads) if allow_leads is not None else os.getenv('AVAMALE_ALLOW_LEADS', '') == '1'
    app = FastAPI(title='AVAMALE Operations Studio', version='0.1.0', docs_url=None, redoc_url=None)
    app.state.store = database
    app.state.failed_logins = {}
    app.state.lead_limits = {}
    app.mount('/assets', StaticFiles(directory=str(APP_DIR / 'static')), name='assets')

    def signature(message: str) -> str:
        return hmac.new(key.encode(), message.encode(), hashlib.sha256).hexdigest()

    def new_token() -> str:
        payload = f'{int(time.time())}.{secrets.token_hex(12)}'
        return payload + '.' + signature('login:' + payload)

    def decode_token(raw: str | None) -> bool:
        if not raw or len(raw) > 180:
            return False
        try:
            timestamp, nonce, digest = raw.split('.')
            stamp = int(timestamp)
            message = f'{timestamp}.{nonce}'
            return (0 <= time.time()-stamp <= SESSION_LIFETIME and
                    bool(re.fullmatch('[0-9a-f]{24}', nonce)) and
                    hmac.compare_digest(digest, signature('login:' + message)))
        except (ValueError, TypeError):
            return False

    def require_auth(request: Request) -> str:
        raw = request.cookies.get(SESSION_COOKIE)
        if not decode_token(raw):
            raise HTTPException(401, detail='Log in to use this workspace')
        return raw

    def require_write(request: Request, x_csrf_token: str | None = Header(default=None)) -> str:
        raw = require_auth(request)
        if not x_csrf_token or not hmac.compare_digest(x_csrf_token, signature('csrf:' + raw)):
            raise HTTPException(403, detail='Missing or invalid CSRF token')
        return raw

    def get_run(identifier: str):
        run = database.get_run(identifier)
        if run is None:
            raise HTTPException(404, detail='Run not found')
        return run

    def export_rows(run):
        for index, finding in enumerate(run['results']):
            decision = run['decisions'][index]
            yield {
                'index': index, 'engine_status': finding.get('status', ''),
                'review_status': decision['state'],
                'record_id': finding.get('record_id') or finding.get('bank_id') or finding.get('shipment_id') or '',
                'finding': finding.get('finding') or finding.get('values') or finding.get('rule') or '',
                'source': finding.get('source_rows') or finding.get('bank_source_row') or finding.get('source_evidence') or '',
                'review_note': decision['note'], 'reviewer': decision['reviewed_by'],
            }

    @app.middleware('http')
    async def secure_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; form-action 'self'"
        return response

    @app.exception_handler(InputError)
    async def bad_input(request: Request, exc: InputError):
        return JSONResponse({'detail': str(exc)}, status_code=400)

    @app.exception_handler(StorageError)
    async def bad_storage(request: Request, exc: StorageError):
        return JSONResponse({'detail': str(exc)}, status_code=400)

    @app.get('/', include_in_schema=False)
    def index():
        return FileResponse(APP_DIR / 'static' / 'index.html')

    @app.get('/products/{code}', include_in_schema=False)
    def product_page(code: str):
        if code not in PRODUCTS:
            raise HTTPException(404, detail='Unknown product')
        return FileResponse(APP_DIR / 'static' / 'index.html')

    @app.get('/healthz')
    def health():
        return {'ok': True, 'mode': 'synthetic-only', 'products': len(PRODUCTS)}

    @app.get('/readyz')
    def readiness():
        """Check that the configured database is reachable and initialized.

        Keep liveness (/healthz) independent of transient database failures.
        No database exception text or secrets are returned to clients.
        """
        try:
            with database.conn() as db:
                db.execute('SELECT COUNT(*) FROM runs').fetchone()
        except Exception:
            return JSONResponse(
                {'ok': False, 'database': 'unavailable'},
                status_code=503,
            )
        return {'ok': True, 'database': 'available'}


    @app.get('/api/products')
    def products():
        return catalog()

    @app.get('/api/products/{code}')
    def product(code: str):
        if code not in PRODUCTS:
            raise HTTPException(404, detail='No product with that code')
        return PRODUCTS[code]

    @app.post('/api/auth/login')
    def login(payload: LoginPayload, request: Request, response: Response):
        client = request.client.host if request.client else 'unknown'
        recent = [t for t in app.state.failed_logins.get(client, []) if time.time()-t < 300]
        if len(recent) >= 10:
            raise HTTPException(429, detail='Too many attempts')
        if not hmac.compare_digest(payload.password.encode(), password.encode()):
            app.state.failed_logins[client] = recent + [time.time()]
            raise HTTPException(401, detail='Invalid credentials')
        app.state.failed_logins.pop(client, None)
        token = new_token()
        response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_LIFETIME, httponly=True,
                            secure=secure_cookie, samesite='strict', path='/')
        return {'authenticated': True, 'csrf': signature('csrf:' + token)}

    @app.post('/api/auth/logout')
    def logout(response: Response, session=Depends(require_write)):
        response.delete_cookie(SESSION_COOKIE, path='/')
        return {'authenticated': False}

    @app.get('/api/auth/me')
    def me(request: Request):
        token = require_auth(request)
        return {'authenticated': True, 'csrf': signature('csrf:' + token)}

    @app.get('/api/samples/{code}/{slot}')
    def sample(code: str, slot: str, session=Depends(require_auth)):
        if code not in PRODUCTS:
            raise HTTPException(404, detail='Unknown product')
        if slot not in PRODUCTS[code]['inputs']:
            raise HTTPException(404, detail='Unknown sample')
        source = APP_DIR.parent / 'samples' / (
            f'{slot}_synthetic.csv' if code == 'AA02' else
            'freight_synthetic.csv' if code == 'AA03' else f'{code}.csv'
        )
        if not source.exists():
            raise HTTPException(404, detail='Sample unavailable')
        return FileResponse(source, filename=source.name, media_type='text/csv')

    @app.post('/api/runs/{code}')
    async def execute(code: str, request: Request, synthetic: Annotated[bool, Form()],
                      data: UploadFile | None = File(default=None),
                      bank: UploadFile | None = File(default=None),
                      ledger: UploadFile | None = File(default=None),
                      date_tolerance: int = Form(default=1),
                      as_of: str = Form(default=''),
                      session=Depends(require_write)):
        if code not in PRODUCTS:
            raise HTTPException(404, detail='Unknown product')
        if request.headers.get('content-length'):
            try:
                if int(request.headers['content-length']) > MAX_UPLOAD * 2 + 40_000:
                    raise HTTPException(413, detail='Request exceeds upload allowance')
            except ValueError:
                raise HTTPException(400, detail='Invalid content length')
        supplied = {'data': data, 'bank': bank, 'ledger': ledger}
        uploads = {}
        for slot, upload in supplied.items():
            if upload is None:
                continue
            if not upload.filename:
                raise HTTPException(400, detail='Upload filename missing')
            raw = await upload.read(MAX_UPLOAD + 1)
            if len(raw) > MAX_UPLOAD:
                raise HTTPException(413, detail=f'{slot} exceeds {MAX_UPLOAD} bytes')
            uploads[slot] = (upload.filename, raw)
        result = run_product(code, uploads, declared_synthetic=synthetic,
                             as_of=as_of or None, date_tolerance=date_tolerance)
        identifier = database.create_run(code, f'{code} · {PRODUCTS[code]["name"]}',
                                         result['hashes'], result['names'], result['findings'])
        return {'id': identifier, 'product': code, 'findings': len(result['findings']),
                'state': 'in_review', 'synthetic': True, 'production_certified': False}

    @app.get('/api/runs')
    def runs(product: str | None = None, session=Depends(require_auth)):
        if product and product not in PRODUCTS:
            raise HTTPException(404, detail='Unknown product')
        return database.list_runs(product=product)

    @app.get('/api/runs/{identifier}')
    def run_details(identifier: str, session=Depends(require_auth)):
        return get_run(identifier)

    @app.post('/api/runs/{identifier}/findings/{index}')
    def review(identifier: str, index: int, payload: DecisionPayload,
               session=Depends(require_write)):
        get_run(identifier)
        return database.update_decision(identifier, index, payload.status,
                                        payload.note, payload.reviewer)

    @app.get('/api/runs/{identifier}/history')
    def history(identifier: str, session=Depends(require_auth)):
        get_run(identifier)
        return database.events(identifier)

    @app.get('/api/runs/{identifier}/export.json')
    def export_json(identifier: str, session=Depends(require_auth)):
        run = get_run(identifier)
        return JSONResponse(run, headers={'Content-Disposition': f'attachment; filename="{run["product"]}-synthetic-{identifier}.json"'})

    @app.get('/api/runs/{identifier}/export.csv')
    def export_csv(identifier: str, session=Depends(require_auth)):
        run = get_run(identifier)
        buf = StringIO()
        keys = ['index', 'engine_status', 'review_status', 'record_id', 'finding', 'source', 'review_note', 'reviewer']
        writer = csv.DictWriter(buf, fieldnames=keys)
        writer.writeheader()
        for row in export_rows(run):
            writer.writerow({k: as_csv_cell(v) for k,v in row.items()})
        return Response(content=buf.getvalue(), media_type='text/csv; charset=utf-8',
                        headers={'Content-Disposition': f'attachment; filename="{run["product"]}-synthetic-{identifier}.csv"'})

    @app.get('/api/runs/{identifier}/export.xlsx')
    def export_xlsx(identifier: str, session=Depends(require_auth)):
        from openpyxl import Workbook
        run = get_run(identifier)
        book = Workbook()
        sheet = book.active
        sheet.title = 'Human review'
        keys = ['index', 'engine_status', 'review_status', 'record_id', 'finding', 'source', 'review_note', 'reviewer']
        sheet.append(keys)
        for row in export_rows(run):
            sheet.append([as_csv_cell(row[k]) for k in keys])
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = sheet.dimensions
        meta = book.create_sheet('Read me')
        for values in [
            ['Mode', 'SYNTHETIC-ONLY / DEMONSTRATION'], ['Product', run['product']],
            ['Run ID', run['id']], ['Human decisions', 'Required'],
            ['Production certified', 'NO'], ['No automated postings', 'TRUE'],
        ]:
            meta.append(values)
        output = BytesIO()
        book.save(output)
        output.seek(0)
        return StreamingResponse(output, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                headers={'Content-Disposition': f'attachment; filename="{run["product"]}-synthetic-{identifier}.xlsx"'})

    @app.post('/api/enquiries')
    def enquiry(payload: EnquiryPayload, request: Request):
        if not allow_enquiries:
            raise HTTPException(503, detail='Enquiry intake is disabled pending privacy and deployment approval')
        if payload.website.strip():
            raise HTTPException(400, detail='Invalid enquiry')
        if payload.product not in PRODUCTS or not payload.consent:
            raise HTTPException(400, detail='Choose a valid product and consent to contact')
        if not re.fullmatch(r'[^\s@]+@[^\s@]+\.[A-Za-z]{2,}', payload.contact_email):
            raise HTTPException(400, detail='Invalid contact email')
        client = request.client.host if request.client else 'unknown'
        recent = [t for t in app.state.lead_limits.get(client, []) if time.time()-t < 3600]
        if len(recent) >= 5:
            raise HTTPException(429, detail='Too many enquiries')
        app.state.lead_limits[client] = recent + [time.time()]
        identifier = database.add_enquiry(payload.product, payload.organization.strip(),
                                           payload.contact_email.strip(), payload.problem.strip())
        return {'id': identifier, 'status': 'new', 'note': 'Enquiry recorded for manual review; no automatic email or acceptance'}

    @app.get('/api/enquiries')
    def list_enquiries(session=Depends(require_auth)):
        return database.enquiries()

    @app.post('/api/enquiries/{identifier}/status')
    def set_enquiry_status(identifier: str, payload: EnquiryStatusPayload,
                           session=Depends(require_write)):
        return database.enquiry_status(identifier, payload.status)

    return app


# Configuration is mandatory, so imports fail closed if secrets are absent.
# Use the documented local server launcher to set env before invoking uvicorn.
if os.getenv('AVAMALE_ADMIN_PASSWORD') and os.getenv('AVAMALE_SESSION_SECRET'):
    app = create_app()
else:
    app = None
