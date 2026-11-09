"""Revenue-critical acceptance tests using exclusively synthetic records."""
import csv
from decimal import Decimal
import tempfile
import unittest
from pathlib import Path

from avamale_review import InputError, read_table, load_transactions, reconcile, load_freight, freight_review, write_report
from avamale_suite.addons import run_addon


class RevenueAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def table(self, filename, header, rows):
        dest = self.root / filename
        with dest.open('w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        return dest

    def tx(self, name, rows):
        return load_transactions(self.table(name, ['id','date','amount','currency','reference'], rows))

    def freight(self, rows):
        p = self.table('shipments.csv', ['id','shipment_id','document_type','document_ref','item_code','quantity','packages','gross_weight_kg','consignee','origin_country'], rows)
        return load_freight(p)

    def test_reconciliation_exact_match_is_suggestion_not_payment(self):
        bank=self.tx('b.csv', [['A','2026-11-01','120.00','NGN','A-1']])
        ledger=self.tx('l.csv', [['L','2026-11-01','120','NGN',' A-1 ']])
        findings=reconcile(bank,ledger)
        self.assertEqual(findings[0]['status'],'SUGGESTED_MATCH')
        self.assertTrue(findings[0]['review_required'])
        self.assertEqual(findings[0]['ledger_ids'],'L')

    def test_reconciliation_missing_reference_rejected_as_match(self):
        bank=self.tx('b.csv', [['A','2026-11-01','120.00','NGN','']])
        ledger=self.tx('l.csv', [['L','2026-11-01','120.00','NGN','']])
        self.assertFalse(any(x['status']=='SUGGESTED_MATCH' for x in reconcile(bank,ledger)))

    def test_reconciliation_duplicate_reference_is_held(self):
        bank=self.tx('b.csv', [['A','2026-11-01','120.00','NGN','A-1']])
        ledger=self.tx('l.csv', [['L1','2026-11-01','120.00','NGN','A-1'],['L2','2026-11-01','120.00','NGN','A-1']])
        self.assertEqual(reconcile(bank,ledger)[0]['status'],'DUPLICATE_OR_AMBIGUOUS')

    def test_reconciliation_mismatched_currency_never_matches(self):
        bank=self.tx('b.csv', [['A','2026-11-01','120.00','NGN','A-1']])
        ledger=self.tx('l.csv', [['L','2026-11-01','120.00','USD','A-1']])
        self.assertFalse(any(x['status']=='SUGGESTED_MATCH' for x in reconcile(bank,ledger)))

    def test_reconciliation_date_boundary(self):
        bank=self.tx('b.csv', [['A','2026-11-01','120.00','NGN','A-1']])
        ledger=self.tx('l.csv', [['L','2026-11-03','120.00','NGN','A-1']])
        self.assertEqual(reconcile(bank,ledger,2)[0]['status'],'SUGGESTED_MATCH')
        self.assertFalse(any(x['status']=='SUGGESTED_MATCH' for x in reconcile(bank,ledger,1)))

    def test_reconciliation_bounded_500_row_pilot(self):
        bank_rows=[(f'B{i:03}', '2026-11-01', str(100+i), 'NGN', f'UNIQUE-{i:03}') for i in range(250)]
        ledger_rows=[(f'L{i:03}', '2026-11-01', str(100+i), 'NGN', f'UNIQUE-{i:03}') for i in range(249)]
        bank=self.tx('b.csv',bank_rows)
        ledger=self.tx('l.csv',ledger_rows)
        findings=reconcile(bank,ledger)
        self.assertEqual(len(bank)+len(ledger),499)
        self.assertEqual(sum(x['status']=='SUGGESTED_MATCH' for x in findings),249)
        self.assertEqual(sum(x['status']=='UNMATCHED_BANK' for x in findings),1)
        self.assertTrue(all(x['review_required'] for x in findings))

    def test_report_rejects_unsafe_formula_and_produces_provenance(self):
        p=self.table('evidence.csv',['id','date','amount','currency','reference'],[['A','2026-11-01','120','NGN','=HYPERLINK(1)']])
        bank=load_transactions(p)
        results=reconcile(bank,[])
        write_report(results,self.root/'out','Synthetic pilot',[p],{'mode':'exact'},True)
        self.assertIn('sha256',(self.root/'out'/'manifest.json').read_text())
        report=(self.root/'out'/'review.csv').read_text(encoding='utf-8-sig')
        self.assertIn("'=HYPERLINK(1)",report)

    def test_freight_detects_discrepant_quantity_not_silently_approved(self):
        rows=self.freight([
          ['I','S','INVOICE','REF1','SKU','50','5','100','Consignee','NG'],
          ['P','S','PACKING_LIST','REF2','SKU','51','5','100','Consignee','NG'],
          ['B','S','BILL_OF_LADING','REF3','SKU','50','5','100','Consignee','NG'],
        ])
        findings=freight_review(rows)
        self.assertIn('quantity',[r['field'] for r in findings if r['status']=='FIELD_DISCREPANCY'])
        self.assertTrue(all(x['review_required'] for x in findings))

    def test_freight_missing_document_is_flagged(self):
        rows=self.freight([
          ['I','S','INVOICE','REF1','SKU','50','5','100','Consignee','NG'],
          ['P','S','PACKING_LIST','REF2','SKU','50','5','100','Consignee','NG']])
        self.assertIn('MISSING_DOCUMENT',[x['status'] for x in freight_review(rows)])

    def test_freight_rejects_invalid_weight(self):
        with self.assertRaises(InputError):
            self.freight([['I','S','INVOICE','REF1','SKU','50','5','-2','Consignee','NG']])







if __name__=='__main__':unittest.main()
