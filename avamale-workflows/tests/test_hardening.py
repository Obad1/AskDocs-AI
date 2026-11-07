"""Cross-product fail-closed regressions; fixtures contain wholly synthetic data."""
import copy
import csv
from datetime import date
from pathlib import Path
import tempfile
import unittest

from avamale_review import InputError, MAX_INPUT_BYTES, MAX_ROWS, load_transactions, read_table
from avamale_suite.workflows import IMPLEMENTATIONS, LINK_FIELDS, RECORD_TYPES, SPEC, run_workflow
from avamale_suite.addons import run_addon

ROOT = Path(__file__).resolve().parents[1]
AS_OF = date(2026, 11, 7)


class HardeningTests(unittest.TestCase):
    def fixture(self, code):
        return read_table(ROOT / 'samples' / f'{code}.csv', SPEC[code][1].split(','))

    def test_each_of_16_adapter_link_fields_rejects_blank_values(self):
        checked = set()
        for code in sorted(IMPLEMENTATIONS):
            with self.subTest(code=code):
                rows = self.fixture(code)
                self.assertTrue(run_workflow(code, rows, AS_OF))
                for field in LINK_FIELDS[code]:
                    modified = copy.deepcopy(rows)
                    modified[0][field] = '   '
                    with self.assertRaisesRegex(InputError, 'linking field'):
                        run_workflow(code, modified, AS_OF)
                    checked.add(code)
        self.assertEqual(len(checked), 16)

    def test_invalid_record_types_fail_closed_across_adapters(self):
        for code in RECORD_TYPES:
            with self.subTest(code=code):
                rows = self.fixture(code)
                rows[0]['record_type'] = 'TYPO_RECORD_TYPE'
                with self.assertRaisesRegex(InputError, 'unsupported record_type'):
                    run_workflow(code, rows, AS_OF)

    def test_aa06_mismatched_transaction_and_amount_never_pass(self):
        rows = self.fixture('AA06')
        self.assertEqual(run_workflow('AA06', rows, AS_OF)[0]['status'], 'REVIEW_PACKET_METADATA')
        for field, value in [('transaction_ref', 'OTHER'), ('amount', '200'), ('currency', 'USD')]:
            with self.subTest(field=field):
                trial = copy.deepcopy(rows)
                trial[1][field] = value
                self.assertEqual(run_workflow('AA06', trial, AS_OF)[0]['status'], 'HOLD_INCOMPLETE_DISPUTE')

    def test_aa08_unknown_urgency_and_empty_request_reference_hold(self):
        rows = self.fixture('AA08')
        trial = copy.deepcopy(rows)
        trial[1]['urgency'] = 'UNMAPPED'
        self.assertEqual(run_workflow('AA08', trial, AS_OF)[0]['status'], 'HOLD_INVALID_URGENCY')
        trial[1]['request_ref'] = ''
        with self.assertRaises(InputError):
            run_workflow('AA08', trial, AS_OF)

    def test_aa09_unknown_opt_out_never_qualifies_for_contact(self):
        for raw in ('UNKNOWN', '', 'maybe'):
            with self.subTest(raw=raw):
                rows=self.fixture('AA09')
                rows[0]['unsubscribed']=raw
                result=run_workflow('AA09', rows, AS_OF)
                self.assertEqual(result[0]['status'], 'HOLD_CONTACT_OR_DATA')
                self.assertTrue(result[0]['review_required'])

    def test_aa14_unknown_auth_and_attachments_request_review(self):
        rows=self.fixture('AA14')
        rows[1]['authentication_result']='UNKNOWN'
        self.assertEqual(run_workflow('AA14', rows, AS_OF)[1]['status'], 'SECURITY_ANALYST_REVIEW')
        rows[1]['authentication_result']='PASS'
        rows[1]['attachments']='2'
        self.assertEqual(run_workflow('AA14', rows, AS_OF)[1]['status'], 'SECURITY_ANALYST_REVIEW')

    def test_aa18_duplicate_or_unverified_evidence_holds(self):
        original = self.fixture('AA18')
        self.assertEqual(run_workflow('AA18', original, AS_OF)[0]['status'], 'REVIEW_PACKET_METADATA')
        trial=copy.deepcopy(original)
        trial[1]['verified']='NO'
        self.assertEqual(run_workflow('AA18', trial, AS_OF)[0]['status'], 'HOLD_MISSING_PACKET_ITEMS')
        trial=copy.deepcopy(original)
        duplicate=copy.copy(trial[1])
        duplicate['id']='E_NEW'
        trial.append(duplicate)
        self.assertEqual(run_workflow('AA18', trial, AS_OF)[0]['status'], 'HOLD_MISSING_PACKET_ITEMS')

    def test_biz_assist_unknown_opt_out_blocks_outreach(self):
        rows = [dict(id='SYN1', sector='finance', pain='Manual reviews', monthly_volume='50', buyer_role='CFO',
                     budget_confirmed='yes', contact_permission='yes', opted_out='unknown')]
        result = run_addon('BIZ_ASSIST', rows)
        self.assertEqual(result[0]['status'], 'DO_NOT_CONTACT')
        self.assertFalse(result[0]['autonomous_action_taken'])

    def test_duplicate_case_and_whitespace_ids_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'ids.csv'
            p.write_text('id,reference\na,one\n A ,two\n', encoding='utf-8')
            with self.assertRaisesRegex(InputError, 'duplicate id'):
                read_table(p,['id','reference'])

    def test_streamed_csv_row_cap_and_file_size_cap(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'many.csv'
            with p.open('w',newline='',encoding='utf-8') as f:
                writer=csv.writer(f)
                writer.writerow(['id','reference'])
                for i in range(MAX_ROWS+1):
                    writer.writerow([i,'SYNTHETIC'])
            with self.assertRaisesRegex(InputError, 'maximum'):
                read_table(p,['id','reference'])
            p.write_text('id,reference\n1,"' + 'x' * (MAX_INPUT_BYTES+2) + '"\n')
            with self.assertRaisesRegex(InputError, 'byte local review limit'):
                read_table(p,['id','reference'])

    def test_malformed_quoted_csv_rejected_before_review(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'malformed.csv'
            for text in ('id,reference\nSYN1,"quoted"junk\n',
                         'id,reference\nSYN1,"never closed\n'):
                with self.subTest(text=text):
                    p.write_text(text, encoding='utf-8')
                    with self.assertRaisesRegex(InputError, 'invalid CSV'):
                        read_table(p, ['id', 'reference'])

    def test_csv_large_column_and_oversized_cell_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'large.csv'
            p.write_text('id,reference\n1,'+'x'*50001+'\n')
            with self.assertRaisesRegex(InputError, 'excessive'):
                read_table(p,['id','reference'])
            p.write_text(','.join(['id','reference']+['col'+str(i) for i in range(63)])+'\n'
                         +','.join(['SYN1','X']+['v']*63)+'\n')
            with self.assertRaisesRegex(InputError, 'excessive'):
                read_table(p,['id','reference'])

if __name__ == '__main__':
    unittest.main()
