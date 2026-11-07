"""Local, human-reviewed AA02/AA03 prototypes. No network, OCR or live integrations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

VERSION = '0.1.0'
MAX_ROWS = 10000
MAX_INPUT_BYTES = 8_000_000
MAX_COLUMNS = 64
MAX_CELL_CHARS = 50_000


class InputError(ValueError):
    pass


def norm(value):
    return ' '.join(str(value or '').strip().split()).upper()


def number(value, label, positive=False):
    text = str(value).strip()
    if not re.fullmatch(r'[+-]?\d+(?:\.\d{1,2})?', text):
        raise InputError(f'{label}: use a decimal with at most 2 places, no commas or currency symbols')
    try:
        n = Decimal(text)
    except InvalidOperation as exc:
        raise InputError(f'{label}: invalid decimal') from exc
    if not n.is_finite() or (positive and n <= 0):
        raise InputError(f'{label}: expected a finite' + (' positive number' if positive else ' number'))
    return n


def read_table(path, required):
    """Read first worksheet or UTF-8 CSV; reject formulas/duplicate IDs and headers."""
    path = Path(path)
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise InputError(f'{path.name}: input exceeds the {MAX_INPUT_BYTES} byte local review limit')
    if path.suffix.lower() == '.csv':
        matrix = []
        with path.open(newline='', encoding='utf-8-sig') as f:
            try:
                for line, row in enumerate(csv.reader(f), 1):
                    if line > MAX_ROWS + 1:
                        raise InputError(f'{path.name}: maximum {MAX_ROWS} rows')
                    if len(row) > MAX_COLUMNS or any(len(cell) > MAX_CELL_CHARS for cell in row):
                        raise InputError(f'{path.name} row {line}: excessive columns or cell length')
                    matrix.append(row)
            except (csv.Error, UnicodeError) as exc:
                raise InputError(f'{path.name}: invalid CSV encoding or quoting') from exc
    elif path.suffix.lower() == '.xlsx':
        from openpyxl import load_workbook
        book = load_workbook(path, read_only=True, data_only=False, keep_links=False)
        try:
            sheet = book.worksheets[0]
            if sheet.max_column > MAX_COLUMNS:
                raise InputError(f'{path.name}: maximum {MAX_COLUMNS} columns')
            matrix = []
            for line, row in enumerate(sheet.iter_rows(), 1):
                if line > MAX_ROWS + 1:
                    raise InputError(f'{path.name}: maximum {MAX_ROWS} rows')
                if any(c.data_type == 'f' for c in row):
                    raise InputError(f'{path.name}: formula cells are not permitted; export values only')
                values = ['' if c.value is None else str(c.value) for c in row]
                if any(len(value) > MAX_CELL_CHARS for value in values):
                    raise InputError(f'{path.name} row {line}: excessive cell length')
                matrix.append(values)
        finally:
            book.close()
    else:
        raise InputError('Only .csv and .xlsx files are supported; no PDF/OCR/macros')
    if not matrix:
        raise InputError(f'{path.name}: empty input')
    headers = [str(x).strip().lower() for x in matrix[0]]
    if any(not h for h in headers) or len(headers) != len(set(headers)):
        raise InputError(f'{path.name}: blank or duplicate column headers')
    missing = set(required) - set(headers)
    if missing:
        raise InputError(f'{path.name}: missing columns {sorted(missing)}')
    if len(matrix) - 1 > MAX_ROWS:
        raise InputError(f'{path.name}: maximum {MAX_ROWS} rows')
    rows, ids = [], set()
    for line, values in enumerate(matrix[1:], 2):
        if not any(str(v).strip() for v in values):
            continue
        if len(values) != len(headers):
            raise InputError(f'{path.name} row {line}: column count differs from header')
        r = dict(zip(headers, values))
        r['_source_row'] = line
        r['_source_file'] = path.name
        unique_id = norm(r.get('id', ''))
        if not unique_id or unique_id in ids:
            raise InputError(f'{path.name} row {line}: blank or duplicate id')
        ids.add(unique_id)
        rows.append(r)
    if not rows:
        raise InputError(f'{path.name}: no data rows')
    return rows


def load_transactions(path):
    rows = read_table(path, ['id', 'date', 'amount', 'currency', 'reference'])
    for r in rows:
        where = f"{r['_source_file']} row {r['_source_row']}"
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(r['date'])):
            raise InputError(f'{where}: date must be YYYY-MM-DD')
        try:
            r['_date'] = date.fromisoformat(r['date'])
        except ValueError as exc:
            raise InputError(f'{where}: invalid calendar date') from exc
        r['_amount'] = number(r['amount'], where + ' amount')
        r['_currency'] = norm(r['currency'])
        if not re.fullmatch(r'[A-Z]{3}', r['_currency']):
            raise InputError(f'{where}: currency must be a three-letter code')
        r['_reference'] = norm(r['reference'])
    return rows


def reconcile(bank, ledger, date_tolerance=0):
    if not isinstance(date_tolerance, int) or not 0 <= date_tolerance <= 7:
        raise InputError('Date tolerance must be an integer from 0 to 7 days')
    buckets = defaultdict(list)
    for j, r in enumerate(ledger):
        buckets[(r['_amount'], r['_currency'], r['_reference'])].append(j)
    candidates, reverse = {}, defaultdict(list)
    for i, b in enumerate(bank):
        js = []
        if b['_reference']:
            for j in buckets[(b['_amount'], b['_currency'], b['_reference'])]:
                if abs((b['_date'] - ledger[j]['_date']).days) <= date_tolerance:
                    js.append(j)
                    reverse[j].append(i)
        candidates[i] = js
    duplicate_key = lambda r: (r['_date'], r['_amount'], r['_currency'], r['_reference'])
    bd = Counter(duplicate_key(r) for r in bank)
    ld = Counter(duplicate_key(r) for r in ledger)
    used, results = set(), []
    for i, b in enumerate(bank):
        js = candidates[i]
        is_dup = bd[duplicate_key(b)] > 1
        if len(js) == 1 and len(reverse[js[0]]) == 1 and not is_dup and ld[duplicate_key(ledger[js[0]])] == 1:
            j = js[0]
            used.add(j)
            results.append({'status': 'SUGGESTED_MATCH', 'bank_id': b['id'], 'ledger_ids': ledger[j]['id'],
                            'bank_source_row': b['_source_row'], 'ledger_source_rows': ledger[j]['_source_row'],
                            'amount': b['_amount'], 'currency': b['_currency'], 'reference': b['reference'],
                            'rule': 'Unique exact amount/currency/normalised-reference; date within tolerance',
                            'review_required': True})
        else:
            status = 'DUPLICATE_OR_AMBIGUOUS' if is_dup or js else ('MISSING_REFERENCE' if not b['_reference'] else 'UNMATCHED_BANK')
            results.append({'status': status, 'bank_id': b['id'], 'ledger_ids': ' | '.join(ledger[j]['id'] for j in js),
                            'bank_source_row': b['_source_row'], 'ledger_source_rows': ' | '.join(str(ledger[j]['_source_row']) for j in js),
                            'amount': b['_amount'], 'currency': b['_currency'], 'reference': b['reference'],
                            'rule': 'No automatic resolution; all candidates retained', 'review_required': True})
    for j, r in enumerate(ledger):
        if j not in used:
            status = 'DUPLICATE_OR_AMBIGUOUS_LEDGER' if reverse[j] or ld[duplicate_key(r)] > 1 else ('MISSING_REFERENCE_LEDGER' if not r['_reference'] else 'UNMATCHED_LEDGER')
            results.append({'status': status, 'bank_id': '', 'ledger_ids': r['id'], 'bank_source_row': '',
                            'ledger_source_rows': r['_source_row'], 'amount': r['_amount'], 'currency': r['_currency'],
                            'reference': r['reference'], 'rule': 'Accountant review required; no posting', 'review_required': True})
    return results


def load_freight(path):
    rows = read_table(path, ['id', 'shipment_id', 'document_type', 'document_ref', 'item_code', 'quantity', 'packages', 'gross_weight_kg', 'consignee', 'origin_country'])
    for r in rows:
        where = f"{r['_source_file']} row {r['_source_row']}"
        r['_type'] = norm(r['document_type'])
        if r['_type'] not in {'INVOICE', 'PACKING_LIST', 'BILL_OF_LADING'}:
            raise InputError(f'{where}: unsupported document_type')
        for f in ['shipment_id', 'document_ref', 'item_code', 'consignee', 'origin_country']:
            if not str(r[f]).strip():
                raise InputError(f'{where}: {f} is required')
        for f in ['quantity', 'packages', 'gross_weight_kg']:
            r['_' + f] = number(r[f], where + ' ' + f, positive=True)
        if r['_packages'] != r['_packages'].to_integral_value():
            raise InputError(f'{where}: packages must be a positive integer')
        if not re.fullmatch(r'[A-Z]{2}', norm(r['origin_country'])):
            raise InputError(f'{where}: origin_country must be a two-letter code')
    return rows


def freight_review(rows, weight_tolerance=Decimal('0')):
    """Compare pre-transcribed, item-level records only; never infer PDF content/customs rules."""
    if not isinstance(weight_tolerance, Decimal) or not weight_tolerance.is_finite() or weight_tolerance < 0:
        raise InputError('Weight tolerance must be a finite nonnegative Decimal')
    groups = defaultdict(list)
    for r in rows:
        groups[(norm(r['shipment_id']), norm(r['item_code']))].append(r)
    results = []
    required = {'INVOICE', 'PACKING_LIST', 'BILL_OF_LADING'}
    fields = ['quantity', 'packages', 'gross_weight_kg', 'consignee', 'origin_country']
    for (shipment, item), group in sorted(groups.items()):
        counts = Counter(r['_type'] for r in group)
        evidence = ' | '.join(f"{r['_type']}:{r['document_ref']}@row{r['_source_row']}" for r in group)
        base = {'shipment_id': shipment, 'item_code': item, 'source_evidence': evidence, 'review_required': True}
        missing = required - set(counts)
        if missing:
            results.append(dict(base, status='MISSING_DOCUMENT', field='document_type', values=' | '.join(sorted(missing)), rule='Required document types absent for this item'))
        duplicates = [k for k, v in counts.items() if v > 1]
        if duplicates:
            results.append(dict(base, status='DUPLICATE_DOCUMENT_ITEM', field='document_type', values=' | '.join(sorted(duplicates)), rule='Duplicate/amended record requires reviewer; no automatic latest-version selection'))
        differences = 0
        for f in fields:
            if f in {'quantity', 'packages', 'gross_weight_kg'}:
                values = [r['_' + f] for r in group]
                same = max(values) - min(values) <= (weight_tolerance if f == 'gross_weight_kg' else Decimal('0'))
            else:
                values = [norm(r[f]) for r in group]
                same = len(set(values)) == 1
            if not same:
                differences += 1
                results.append(dict(base, status='FIELD_DISCREPANCY', field=f,
                                    values=' | '.join(f"{r['_type']}={r[f]}" for r in group),
                                    rule='Explicit field comparison; units/meaning must be agreed by reviewer'))
        if not missing and not duplicates and not differences:
            results.append(dict(base, status='CONSISTENT_FIELDS_REVIEW_REQUIRED', field='', values='', rule='Compared fields agree; not evidence documents are authentic, complete or customs-compliant'))
    return results


def safe_cell(value):
    """Prevent spreadsheet formula injection in untrusted text while retaining original JSON evidence."""
    if isinstance(value, (Decimal, int, bool)):
        return str(value)
    value = str(value)
    if value.lstrip().startswith(('=', '+', '-', '@', '\t', '\r', '\n')) or value.startswith(('\t', '\r', '\n')):
        return "'" + value
    return value


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_report(results, output, title, input_paths, rules, synthetic=False):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    # A new directory is required to avoid accidental overwrite of a prior review.
    if any((out / n).exists() for n in ['review.csv', 'review.xlsx', 'review.json', 'review.html', 'manifest.json']):
        raise InputError('Output already contains review files; choose a new output directory')
    fields = list(results[0]) if results else ['status']
    manifest = {'prototype_version': VERSION, 'title': title, 'synthetic_only_demo': synthetic,
                'processing': 'Local Python process on the machine running this command; no network calls in this application',
                'storage': 'Inputs remain at their original locations; outputs persist in the selected output directory until manually deleted. No automatic secure deletion or encryption provided.',
                'inputs': [{'filename': Path(p).name, 'sha256': fingerprint(p)} for p in input_paths],
                'rules': rules, 'record_counts': dict(Counter(r['status'] for r in results)),
                'human_review_required': True, 'limitations': ['Not a production integration', 'No OCR/PDF extraction', 'No autonomous financial/customs/legal decisions', 'No independently audited security or customer-outcome claims']}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (out / 'review.json').write_text(json.dumps(results, default=str, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out / 'review.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: safe_cell(v) for k, v in r.items()} for r in results)
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    wb = Workbook()
    ws = wb.active
    ws.title = 'Human review'
    ws.append(fields)
    for r in results:
        ws.append([safe_cell(r.get(k, '')) for k in fields])
    ws.freeze_panes = 'A2'
    ws.auto_filter.ref = ws.dimensions
    for c in ws[1]:
        c.font = Font(bold=True, color='FFFFFF')
        c.fill = PatternFill('solid', fgColor='143B45')
    for col in ws.columns:
        from openpyxl.utils import get_column_letter
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(55, max(16, max(len(str(c.value or '')) for c in col) + 2))
        for c in col:
            c.alignment = Alignment(vertical='top', wrap_text=True)
    info = wb.create_sheet('Read first')
    for k, v in manifest.items():
        info.append([k, safe_cell(json.dumps(v, default=str, ensure_ascii=False) if isinstance(v, (dict, list)) else v)])
    info.column_dimensions['A'].width = 26
    info.column_dimensions['B'].width = 110
    for line in info:
        for cell in line:
            cell.alignment = Alignment(wrap_text=True, vertical='top')
    wb.save(out / 'review.xlsx')
    esc = lambda x: html.escape(str(x), quote=True)
    headers = ''.join(f'<th>{esc(f.replace("_", " "))}</th>' for f in fields)
    trs = ''.join('<tr>' + ''.join(f'<td>{esc(r.get(f,""))}</td>' for f in fields) + '</tr>' for r in results)
    cards = ''.join(f'<div class="metric"><strong>{n}</strong><span>{esc(s.replace("_", " "))}</span></div>' for s, n in manifest['record_counts'].items())
    label = 'SYNTHETIC DATA DEMONSTRATION' if synthetic else 'UNVALIDATED PROTOTYPE — REVIEW REQUIRED'
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(title)}</title>
<style>body{{font:15px/1.55 system-ui,sans-serif;margin:0;color:#173a43;background:#f1f6f6}}main{{max-width:1400px;margin:auto;padding:35px}}.brand{{font-weight:800;letter-spacing:.16em}}h1{{font-size:34px;line-height:1.15;max-width:800px}}.label{{font-size:12px;font-weight:800;letter-spacing:.1em;color:#8a4516}}.callout{{background:#fff1dc;border-left:4px solid #c97721;padding:15px 20px;border-radius:5px}}.metrics{{display:flex;flex-wrap:wrap;gap:12px;margin:24px 0}}.metric{{background:white;border:1px solid #d5e2e2;border-radius:10px;padding:15px;min-width:150px;flex:1}}strong{{display:block;font-size:30px}}.metric span{{font-size:11px;letter-spacing:.04em}}.table{{overflow:auto;background:white;border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;vertical-align:top;padding:13px;border-bottom:1px solid #dce6e6;min-width:105px}}th{{background:#173a43;color:white;position:sticky;top:0}}td:first-child{{font-weight:700}}footer{{font-size:12px;margin-top:25px;color:#4f6a71}}details{{background:white;border-radius:8px;padding:16px;margin-top:20px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}@media(max-width:600px){{main{{padding:18px}}h1{{font-size:27px}}}}</style>
<main><div class="brand">AVAMALE / REVIEW LAB</div><p class="label">{label}</p><h1>{esc(title)}</h1><p>Explicit rules. Traceable evidence. Every decision stays with a human reviewer.</p><div class="callout">These are review suggestions, not completed reconciliation, document authentication or professional decisions. No customer outcomes are claimed.</div><section class="metrics">{cards}</section><div class="table"><table><thead><tr>{headers}</tr></thead><tbody>{trs}</tbody></table></div><details><summary>Rules, processing and evidence manifest</summary><pre>{esc(json.dumps(manifest,indent=2))}</pre></details><footer>Prototype {VERSION} · Report contains its displayed data and must be handled accordingly. No external scripts, fonts, analytics or network requests. Generated locally.</footer></main></html>'''
    (out / 'review.html').write_text(page, encoding='utf-8')
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('reconcile', help='Compare structured bank and ledger exports')
    r.add_argument('--bank', required=True)
    r.add_argument('--ledger', required=True)
    r.add_argument('--date-tolerance', type=int, default=0)
    r.add_argument('--out', required=True)
    r.add_argument('--synthetic', action='store_true', help='Label known synthetic demonstrations only')
    f = sub.add_parser('freight', help='Compare pre-transcribed, item-level freight records')
    f.add_argument('--documents', required=True)
    f.add_argument('--weight-tolerance', default='0')
    f.add_argument('--out', required=True)
    f.add_argument('--synthetic', action='store_true')
    a = p.parse_args(argv)
    try:
        if a.command == 'reconcile':
            bank, ledger = load_transactions(a.bank), load_transactions(a.ledger)
            result = reconcile(bank, ledger, a.date_tolerance)
            manifest = write_report(result, a.out, 'AA02 · Reconciliation exception review', [a.bank, a.ledger],
                                    {'date_tolerance_days': a.date_tolerance, 'amount_tolerance': '0', 'matching': 'One-to-one unique exact amount/currency/normalised reference; explicit date window', 'missing_reference': 'Never auto-matched', 'split_payments': 'Not supported'}, a.synthetic)
        else:
            rows = load_freight(a.documents)
            tolerance = number(a.weight_tolerance, 'weight tolerance')
            result = freight_review(rows, tolerance)
            manifest = write_report(result, a.out, 'AA03 · Freight document discrepancy review', [a.documents],
                                    {'gross_weight_tolerance_kg': str(tolerance), 'other_numeric_tolerance': '0', 'comparison_level': 'Each document contains an agreed item-level row; whole-shipment B/L is not comparable without reviewer preparation'}, a.synthetic)
        print(json.dumps({'output_directory': str(Path(a.out).resolve()), 'counts': manifest['record_counts'], 'human_review_required': True}, indent=2))
        return 0
    except (InputError, OSError, ValueError) as exc:
        p.exit(2, f'Input/review error: {exc}\n')


if __name__ == '__main__':
    main()
