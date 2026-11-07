"""Minimal sandbox runner for all 18 distinct, human-review engines."""
from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path
import tempfile

from avamale_review import (InputError, read_table, load_transactions,
                            reconcile, load_freight, freight_review)
from avamale_suite.workflows import SPEC, SENSITIVE_SYNTHETIC_ONLY, run_workflow

MAX_UPLOAD = 2 * 1024 * 1024
MAX_COMBINED_RECORDS = 5000  # capped below the underlying 10k-per-file input parser


def run_product(product: str, uploads: dict[str, tuple[str, bytes]], *,
                declared_synthetic: bool, as_of: str | None = None,
                date_tolerance: int = 1) -> dict:
    if product not in SPEC:
        raise InputError('Unknown product')
    # The entire UI version is demo-only until security and domain sign-off.
    if declared_synthetic is not True:
        raise InputError('The pilot console accepts invented synthetic data ONLY; real customer records are prohibited')
    expected = {'bank', 'ledger'} if product == 'AA02' else {'data'}
    if set(uploads) != expected:
        raise InputError('Incorrect upload slots; expected: ' + ', '.join(sorted(expected)))
    hashes, source_names = {}, {}
    with tempfile.TemporaryDirectory(prefix='avamale-demo-') as temp:
        paths: dict[str, Path] = {}
        for slot, (filename, raw) in uploads.items():
            name = Path(filename).name
            if not name or Path(name).suffix.lower() not in {'.csv', '.xlsx'}:
                raise InputError('Only CSV and values-only XLSX exports are supported')
            if not raw or len(raw) > MAX_UPLOAD:
                raise InputError(f'{slot}: upload must be between 1 byte and 2 MiB')
            path = Path(temp) / (slot + Path(name).suffix.lower())
            path.write_bytes(raw)
            paths[slot] = path
            hashes[slot] = hashlib.sha256(raw).hexdigest()
            source_names[slot] = name[:100]
        if product == 'AA02':
            bank = load_transactions(paths['bank'])
            ledger = load_transactions(paths['ledger'])
            if len(bank) + len(ledger) > MAX_COMBINED_RECORDS:
                raise InputError(f'AA02: combined records exceed {MAX_COMBINED_RECORDS}')
            findings = reconcile(bank, ledger, date_tolerance)
        elif product == 'AA03':
            rows = load_freight(paths['data'])
            if len(rows) > MAX_COMBINED_RECORDS:
                raise InputError(f'AA03: records exceed {MAX_COMBINED_RECORDS}')
            findings = freight_review(rows)
        else:
            fields = SPEC[product][1].split(',')
            rows = read_table(paths['data'], fields)
            if len(rows) > MAX_COMBINED_RECORDS:
                raise InputError(f'{product}: records exceed {MAX_COMBINED_RECORDS}')
            try:
                evaluation_date = date.fromisoformat(as_of or date.today().isoformat())
            except ValueError as exc:
                raise InputError('Invalid evaluation date') from exc
            findings = run_workflow(product, rows, evaluation_date)
    if not findings or any(f.get('review_required') is not True for f in findings):
        raise InputError('Engine did not return nonempty human-review findings')
    return {
        'product': product, 'findings': findings, 'hashes': hashes,
        'names': source_names, 'synthetic': True,
        'human_review_required': True, 'production_certified': False,
        'sensitive': product in SENSITIVE_SYNTHETIC_ONLY,
    }
