"""Durable local case-management for synthetic, human-reviewed demonstrations.

No raw input files are persisted. SQLite is single-instance only; use a managed,
security-reviewed database before customer-facing multi-user operation.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


class StorageError(ValueError):
    pass


class Store:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS runs (
              id TEXT PRIMARY KEY, product TEXT NOT NULL, title TEXT NOT NULL,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              input_hashes TEXT NOT NULL, results TEXT NOT NULL,
              source_names TEXT NOT NULL, synthetic INTEGER NOT NULL CHECK(synthetic=1),
              state TEXT NOT NULL DEFAULT 'in_review',
              CHECK(state IN ('in_review','review_complete'))
            );
            CREATE INDEX IF NOT EXISTS ix_runs_product_created ON runs(product, created_at DESC);
            CREATE TABLE IF NOT EXISTS decisions (
              run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
              finding_index INTEGER NOT NULL CHECK(finding_index>=0),
              state TEXT NOT NULL CHECK(state IN ('pending','accepted','rejected','needs_info')),
              note TEXT NOT NULL DEFAULT '', reviewed_by TEXT NOT NULL DEFAULT '',
              updated_at TEXT NOT NULL,
              PRIMARY KEY(run_id,finding_index)
            );
            CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
              finding_index INTEGER, action TEXT NOT NULL, actor TEXT NOT NULL,
              detail TEXT NOT NULL, at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS enquiries (
              id TEXT PRIMARY KEY, product TEXT NOT NULL,
              organization TEXT NOT NULL, contact_email TEXT NOT NULL,
              problem TEXT NOT NULL, consent INTEGER NOT NULL CHECK(consent=1),
              status TEXT NOT NULL DEFAULT 'new', created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              CHECK(status IN ('new','qualified','proposal','closed'))
            );
            ''')

    @contextmanager
    def conn(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA busy_timeout=10000')
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def create_run(self, product: str, title: str, input_hashes: dict, names: dict, findings: list) -> str:
        identifier = str(uuid.uuid4())
        timestamp = now()
        with self.conn() as db:
            db.execute('''INSERT INTO runs(id,product,title,created_at,updated_at,input_hashes,results,source_names,synthetic)
                          VALUES (?,?,?,?,?,?,?,?,1)''', (identifier, product, title, timestamp, timestamp,
                          json.dumps(input_hashes), json.dumps(findings, default=str), json.dumps(names)))
            for index in range(len(findings)):
                db.execute('INSERT INTO decisions VALUES (?,?,?,?,?,?)', (identifier, index, 'pending', '', '', timestamp))
            db.execute('INSERT INTO events(run_id,finding_index,action,actor,detail,at) VALUES (?,?,?,?,?,?)',
                       (identifier, None, 'created', 'operator', 'Synthetic run created; automated decisions disabled', timestamp))
        return identifier

    def get_run(self, identifier: str) -> dict | None:
        with self.conn() as db:
            row = db.execute('SELECT * FROM runs WHERE id=?', (identifier,)).fetchone()
            if row is None:
                return None
            record = dict(row)
            record['results'] = json.loads(record['results'])
            record['input_hashes'] = json.loads(record['input_hashes'])
            record['source_names'] = json.loads(record['source_names'])
            decisions = db.execute('SELECT * FROM decisions WHERE run_id=? ORDER BY finding_index', (identifier,)).fetchall()
            record['decisions'] = [dict(x) for x in decisions]
            record['synthetic'] = bool(record['synthetic'])
            return record

    def list_runs(self, product: str | None = None, limit: int = 60) -> list[dict]:
        with self.conn() as db:
            if product:
                rows = db.execute('SELECT id,product,title,created_at,updated_at,state FROM runs WHERE product=? ORDER BY created_at DESC LIMIT ?', (product, limit)).fetchall()
            else:
                rows = db.execute('SELECT id,product,title,created_at,updated_at,state FROM runs ORDER BY created_at DESC LIMIT ?', (limit,)).fetchall()
            return [dict(row) for row in rows]

    def update_decision(self, identifier: str, index: int, state: str, note: str, actor: str):
        if state not in {'accepted','rejected','needs_info','pending'}:
            raise StorageError('Unsupported review status')
        if not actor or len(actor) > 80 or len(note) > 2000:
            raise StorageError('Reviewer and note must be bounded')
        with self.conn() as db:
            row = db.execute('SELECT results FROM runs WHERE id=?', (identifier,)).fetchone()
            if row is None:
                raise StorageError('Run not found')
            findings = json.loads(row['results'])
            if not isinstance(index, int) or index < 0 or index >= len(findings):
                raise StorageError('Finding index outside this run')
            timestamp = now()
            db.execute('''UPDATE decisions SET state=?, note=?, reviewed_by=?, updated_at=?
                          WHERE run_id=? AND finding_index=?''', (state, note, actor, timestamp, identifier, index))
            incomplete = db.execute('SELECT COUNT(*) AS total FROM decisions WHERE run_id=? AND state IN ("pending","needs_info")', (identifier,)).fetchone()['total']
            new_state = 'review_complete' if incomplete == 0 else 'in_review'
            db.execute('UPDATE runs SET updated_at=?, state=? WHERE id=?', (timestamp, new_state, identifier))
            db.execute('INSERT INTO events(run_id,finding_index,action,actor,detail,at) VALUES (?,?,?,?,?,?)',
                       (identifier, index, 'decision', actor, json.dumps({'decision': state, 'note': note}), timestamp))
        return {'run_id': identifier, 'index': index, 'decision': state, 'state': new_state}

    def events(self, identifier: str) -> list[dict]:
        with self.conn() as db:
            return [dict(r) for r in db.execute('SELECT id,finding_index,action,actor,detail,at FROM events WHERE run_id=? ORDER BY id', (identifier,)).fetchall()]

    def add_enquiry(self, product: str, organization: str, email: str, problem: str) -> str:
        identifier = str(uuid.uuid4())
        timestamp = now()
        with self.conn() as db:
            db.execute('''INSERT INTO enquiries(id,product,organization,contact_email,problem,consent,created_at,updated_at)
                          VALUES (?,?,?,?,?,1,?,?)''', (identifier, product, organization, email, problem, timestamp, timestamp))
        return identifier

    def enquiries(self) -> list[dict]:
        with self.conn() as db:
            return [dict(r) for r in db.execute('SELECT * FROM enquiries ORDER BY created_at DESC LIMIT 100').fetchall()]

    def enquiry_status(self, identifier: str, state: str):
        if state not in {'new','qualified','proposal','closed'}:
            raise StorageError('Invalid enquiry state')
        with self.conn() as db:
            timestamp = now()
            cursor = db.execute('UPDATE enquiries SET status=?,updated_at=? WHERE id=?', (state, timestamp, identifier))
            if cursor.rowcount != 1:
                raise StorageError('Enquiry not found')
            return {'id': identifier, 'status': state}
