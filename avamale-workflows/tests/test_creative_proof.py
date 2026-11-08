"""Proof that AA01-AA04 creative story fixtures match real synthetic engines."""
import unittest
from datetime import date
from pathlib import Path
from avamale_review import load_freight, freight_review, load_transactions, reconcile, read_table
from avamale_suite.workflows import run_workflow, SPEC

SAMPLES=Path(__file__).resolve().parents[1]/'samples'/'creative_proof'

class CreativeProofTests(unittest.TestCase):
    def test_aa01_approved_and_missing_evidence(self):
        r=run_workflow('AA01',read_table(SAMPLES/'AA01_evidence_story.csv',SPEC['AA01'][1].split(',')),date(2026,11,8))
        statuses={x['status'] for x in r}
        self.assertIn('DRAFT_EVIDENCE_AVAILABLE',statuses)
        self.assertTrue(any(s.startswith('HOLD') for s in statuses))
        self.assertTrue(all(x['review_required'] for x in r))

    def test_aa02_21500_unmatched_not_posted(self):
        b=load_transactions(SAMPLES/'AA02_bank_story.csv')
        l=load_transactions(SAMPLES/'AA02_ledger_story.csv')
        r=reconcile(b,l,0)
        self.assertTrue(any(x['status']=='UNMATCHED_BANK' and x.get('bank_id')=='B-01' for x in r))
        self.assertTrue(any(x['status']=='SUGGESTED_MATCH' for x in r))
        self.assertTrue(all(x['review_required'] for x in r))

    def test_aa03_freight_fields_not_pdf_or_dates(self):
        r=freight_review(load_freight(SAMPLES/'AA03_freight_story.csv'))
        d={x['field'] for x in r if x['status']=='FIELD_DISCREPANCY'}
        self.assertIn('quantity',d)
        self.assertIn('consignee',d)
        self.assertNotIn('date',d)
        self.assertTrue(all(x['review_required'] for x in r))

    def test_aa04_100_100_70_holds(self):
        rows=read_table(SAMPLES/'AA04_payables_story.csv',SPEC['AA04'][1].split(','))
        r=run_workflow('AA04',rows,date(2026,11,8))
        self.assertEqual([x['status'] for x in r],['HOLD_THREE_WAY_EXCEPTION'])
        self.assertTrue(all(x['review_required'] and not x['machine_action_taken'] for x in r))

if __name__=='__main__':
    unittest.main()
