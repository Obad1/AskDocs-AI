"""AVAMALE published examples: synthetic-only AA03 and AA04 pilot checks.

These are structured-record comparisons, not PDF/OCR, real-file or customs decisions.
"""
import csv
import tempfile
import unittest
from datetime import date
from pathlib import Path

from avamale_review import load_freight, freight_review, read_table
from avamale_suite.workflows import run_workflow, SPEC


class PublishedOfferPilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def write(self, name, headers, rows):
        path = self.folder / name
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        return path

    def test_freight_500_vs_480_and_name_spelling(self):
        headers=['id','shipment_id','document_type','document_ref',
                 'item_code','quantity','packages','gross_weight_kg',
                 'consignee','origin_country']
        rows=[
            ['INV','FX-041','INVOICE','INV-FX-041','STATIONERY','500','25','1000.00','Harborfield Retail Ltd','NG'],
            ['PL','FX-041','PACKING_LIST','PL-FX-041','STATIONERY','480','24','960.00','Harborfield Retail Ltd','NG'],
            ['BL','FX-041','BILL_OF_LADING','BL-FX-041','STATIONERY','480','24','960.00','Harbourfield Retail Ltd','NG'],
        ]
        parsed = load_freight(self.write('freight_fictional.csv',headers,rows))
        findings = freight_review(parsed)
        fields={x['field'] for x in findings if x['status']=='FIELD_DISCREPANCY'}
        self.assertIn('quantity',fields)
        self.assertIn('consignee',fields)
        self.assertTrue(all(x['review_required'] for x in findings))
        # Deliberately NOT asserting a date finding: the current AA03 schema
        # does not contain a document revision date field.
        self.assertNotIn('date',fields)

    def test_freight_identical_records_do_not_claim_authenticity(self):
        headers=['id','shipment_id','document_type','document_ref','item_code',
                 'quantity','packages','gross_weight_kg','consignee','origin_country']
        rows=[
            [record,'FX-042',kind,record,'SKU','480','24','960','Example Buyer','NG']
            for record,kind in [('A','INVOICE'),('B','PACKING_LIST'),('C','BILL_OF_LADING')]
        ]
        findings=freight_review(load_freight(self.write('consistent.csv',headers,rows)))
        self.assertEqual([x['status'] for x in findings],['CONSISTENT_FIELDS_REVIEW_REQUIRED'])

    def test_payables_100_invoiced_70_received(self):
        headers=SPEC['AA04'][1].split(',')
        rows=[
            ['PO1','PO','PO-FX-041','SKU','SYNTHETIC SUPPLIER','100','200','NGN'],
            ['INV1','INVOICE','PO-FX-041','SKU','SYNTHETIC SUPPLIER','100','200','NGN'],
            ['REC1','RECEIPT','PO-FX-041','SKU','SYNTHETIC SUPPLIER','70','200','NGN'],
        ]
        parsed=read_table(self.write('payables_fictional.csv',headers,rows),headers)
        result=run_workflow('AA04',parsed,date(2026,11,8))
        self.assertEqual([x['status'] for x in result],['HOLD_THREE_WAY_EXCEPTION'])
        self.assertTrue(result[0]['review_required'])
        self.assertFalse(result[0]['machine_action_taken'])

    def test_payables_three_equal_quantities_still_require_review(self):
        headers=SPEC['AA04'][1].split(',')
        rows=[
            ['PO','PO','PO2','SKU','SYNTHETIC','100','200','NGN'],
            ['INV','INVOICE','PO2','SKU','SYNTHETIC','100','200','NGN'],
            ['GR','RECEIPT','PO2','SKU','SYNTHETIC','100','200','NGN'],
        ]
        parsed=read_table(self.write('payables_equal.csv',headers,rows),headers)
        result=run_workflow('AA04',parsed,date(2026,11,8))
        self.assertEqual(result[0]['status'],'REVIEW_THREE_WAY_AGREEMENT')
        self.assertTrue(result[0]['review_required'])


if __name__=='__main__':
    unittest.main()
