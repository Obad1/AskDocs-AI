import csv
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from openpyxl import Workbook, load_workbook

from avamale_review import (InputError, load_transactions, reconcile, load_freight,
                            freight_review, write_report, safe_cell, number)

ROOT = Path(__file__).resolve().parents[1]


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def transactions(self, name, rows):
        p = self.root / name
        with p.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['id', 'date', 'amount', 'currency', 'reference'])
            w.writerows(rows)
        return load_transactions(p)

    def test_exact_decimal_match(self):
        a = self.transactions('a.csv', [['A', '2026-01-01', '0.10', 'NGN', ' Ref 1 ']])
        b = self.transactions('b.csv', [['B', '2026-01-01', '0.1', 'NGN', 'ref 1']])
        r = reconcile(a, b)
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0]['status'], 'SUGGESTED_MATCH')
        self.assertEqual(r[0]['bank_source_row'], 2)
        self.assertTrue(r[0]['review_required'])

    def test_date_tolerance_is_explicit(self):
        a = self.transactions('a.csv', [['A', '2026-01-01', '12.00', 'NGN', 'X']])
        b = self.transactions('b.csv', [['B', '2026-01-02', '12.00', 'NGN', 'X']])
        self.assertFalse(any(r['status'] == 'SUGGESTED_MATCH' for r in reconcile(a, b)))
        self.assertEqual(reconcile(a, b, 1)[0]['status'], 'SUGGESTED_MATCH')
        with self.assertRaises(InputError):
            reconcile(a, b, 8)

    def test_currency_and_amount_mismatch_never_match(self):
        a = self.transactions('a.csv', [['A', '2026-01-01', '12.00', 'USD', 'X']])
        for currency, amount in [('NGN', '12.00'), ('USD', '12.01')]:
            b = self.transactions('b.csv', [['B', '2026-01-01', amount, currency, 'X']])
            self.assertFalse(any(r['status'] == 'SUGGESTED_MATCH' for r in reconcile(a, b)))

    def test_duplicate_bank_rows_not_silently_deduplicated(self):
        a = self.transactions('a.csv', [['A1', '2026-01-01', '12', 'NGN', 'X'], ['A2', '2026-01-01', '12', 'NGN', 'X']])
        b = self.transactions('b.csv', [['B', '2026-01-01', '12', 'NGN', 'X']])
        r = reconcile(a, b)
        self.assertEqual([x['status'] for x in r], ['DUPLICATE_OR_AMBIGUOUS', 'DUPLICATE_OR_AMBIGUOUS', 'DUPLICATE_OR_AMBIGUOUS_LEDGER'])
        self.assertEqual({x['bank_id'] for x in r if x['bank_id']}, {'A1', 'A2'})

    def test_duplicate_ledger_rows_not_arbitrarily_selected(self):
        a = self.transactions('a.csv', [['A', '2026-01-01', '12', 'NGN', 'X']])
        b = self.transactions('b.csv', [['B1', '2026-01-01', '12', 'NGN', 'X'], ['B2', '2026-01-01', '12', 'NGN', 'X']])
        r = reconcile(a, b)
        self.assertEqual(r[0]['ledger_ids'], 'B1 | B2')
        self.assertFalse(any(x['status'] == 'SUGGESTED_MATCH' for x in r))

    def test_missing_references_never_auto_match(self):
        a = self.transactions('a.csv', [['A', '2026-01-01', '12', 'NGN', '']])
        b = self.transactions('b.csv', [['B', '2026-01-01', '12', 'NGN', '']])
        self.assertEqual([r['status'] for r in reconcile(a, b)], ['MISSING_REFERENCE', 'MISSING_REFERENCE_LEDGER'])

    def test_duplicate_ids_and_bad_dates_rejected(self):
        with self.assertRaises(InputError):
            self.transactions('a.csv', [['A', '2026-01-01', '12', 'NGN', 'X'], ['A', '2026-01-01', '12', 'NGN', 'X']])
        with self.assertRaises(InputError):
            self.transactions('a.csv', [['A', '2026-02-30', '12', 'NGN', 'X']])

    def test_nan_infinite_or_overprecise_amounts_rejected(self):
        for value in ['NaN', 'Infinity', '1.001', '1,000', '1e3']:
            with self.subTest(value=value), self.assertRaises(InputError):
                number(value, 'amount')

    def test_xlsx_values_supported_but_formulas_rejected(self):
        p = self.root / 'a.xlsx'
        w = Workbook()
        s = w.active
        s.append(['id', 'date', 'amount', 'currency', 'reference'])
        s.append(['A', '2026-01-01', 12.5, 'NGN', 'X'])
        w.save(p)
        self.assertEqual(load_transactions(p)[0]['_amount'], Decimal('12.5'))
        s['E2'] = '=1+1'
        w.save(p)
        with self.assertRaises(InputError):
            load_transactions(p)

    def test_reports_escape_html_formula_text_and_preserve_json(self):
        r = [{'status': 'UNMATCHED_BANK', 'reference': '=1+1', 'note': '<script>alert(1)</script>', 'review_required': True}]
        out = self.root / 'reports'
        write_report(r, out, 'Synthetic test', [ROOT / 'samples/bank_synthetic.csv'], {}, True)
        self.assertEqual(json.loads((out / 'review.json').read_text())[0]['reference'], '=1+1')
        self.assertIn("'=1+1", (out / 'review.csv').read_text(encoding='utf-8-sig'))
        page = (out / 'review.html').read_text()
        self.assertNotIn('<script>', page)
        self.assertIn('&lt;script&gt;', page)
        w = load_workbook(out / 'review.xlsx', data_only=False)
        self.assertNotEqual(w.active['B2'].data_type, 'f')
        w.close()
        with self.assertRaises(InputError):
            write_report(r, out, 'test', [ROOT / 'samples/bank_synthetic.csv'], {})

    def test_formula_injection_whitespace(self):
        for text in ['=X', '+X', '-X', '@X', '   =X', '\tX', '\rX']:
            self.assertTrue(safe_cell(text).startswith("'"))
        self.assertEqual(safe_cell(Decimal('-12.00')), '-12.00')

    def test_synthetic_reconciliation_expected_outcomes(self):
        a = load_transactions(ROOT / 'samples/bank_synthetic.csv')
        b = load_transactions(ROOT / 'samples/ledger_synthetic.csv')
        r = reconcile(a, b, 1)
        self.assertEqual({x['bank_id'] for x in r if x['status'] == 'SUGGESTED_MATCH'}, {'B001', 'B002'})
        self.assertEqual(len([x for x in r if x['bank_id']]), len(a))
        matched_l = {x['ledger_ids'] for x in r if x['status'] == 'SUGGESTED_MATCH'}
        unmatched_l = {x['ledger_ids'] for x in r if not x['bank_id']}
        self.assertEqual(matched_l | unmatched_l, {x['id'] for x in b})
        self.assertFalse(matched_l & unmatched_l)

    def test_freight_expected_discrepancies(self):
        r = freight_review(load_freight(ROOT / 'samples/freight_synthetic.csv'))
        self.assertEqual([x['status'] for x in r if x['shipment_id'] == 'S001'], ['CONSISTENT_FIELDS_REVIEW_REQUIRED'])
        s2 = {x['field'] for x in r if x['shipment_id'] == 'S002'}
        self.assertEqual(s2, {'quantity', 'packages', 'gross_weight_kg', 'consignee', 'origin_country'})
        self.assertEqual([x['status'] for x in r if x['shipment_id'] == 'S003'], ['MISSING_DOCUMENT'])
        self.assertIn('DUPLICATE_DOCUMENT_ITEM', [x['status'] for x in r if x['shipment_id'] == 'S004'])

    def test_weight_tolerance_changes_only_weight_comparison(self):
        r = freight_review(load_freight(ROOT / 'samples/freight_synthetic.csv'), Decimal('10'))
        self.assertNotIn('gross_weight_kg', {x['field'] for x in r if x['shipment_id'] == 'S002'})
        self.assertIn('quantity', {x['field'] for x in r if x['shipment_id'] == 'S002'})
        for bad in [Decimal('-1'), Decimal('NaN')]:
            with self.assertRaises(InputError):
                freight_review([], bad)

    def test_freight_requires_item_level_schema(self):
        p = self.root / 'bad.csv'
        p.write_text('id,shipment_id,document_type\nA,S1,INVOICE\n')
        with self.assertRaises(InputError):
            load_freight(p)


if __name__ == '__main__':
    unittest.main()
