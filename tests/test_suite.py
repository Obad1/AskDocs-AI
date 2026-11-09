"""Cross-workflow validation and privacy regression checks. All test data are synthetic."""
import copy
from datetime import date
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from avamale_review import InputError, read_table
from avamale_suite.feedback import add_feedback, delete_feedback, summarize, set_feedback_status
from avamale_suite.workflows import SPEC, IMPLEMENTATIONS, SENSITIVE_SYNTHETIC_ONLY, run_workflow

ROOT = Path(__file__).resolve().parents[1]
AS_OF = date(2026, 11, 7)


class WorkflowSuiteTests(unittest.TestCase):
    def fixture(self, code):
        return read_table(ROOT / 'samples' / f'{code}.csv', SPEC[code][1].split(','))

    def test_all_18_registered_and_16_separate_adapters(self):
        self.assertEqual(set(SPEC), {f'AA{i:02d}' for i in range(1, 19)})
        self.assertEqual(set(IMPLEMENTATIONS), set(SPEC) - {'AA02', 'AA03'})
        self.assertEqual(len(set(IMPLEMENTATIONS.values())), 16)

    def test_all_16_adapter_fixtures_have_human_review_findings(self):
        for code in IMPLEMENTATIONS:
            with self.subTest(workflow=code):
                findings = run_workflow(code, self.fixture(code), AS_OF)
                self.assertTrue(findings)
                for finding in findings:
                    self.assertTrue(finding['review_required'])
                    self.assertFalse(finding['machine_action_taken'])
                    self.assertIn('source_rows', finding)
                    self.assertIn('recommended_human_action', finding)
                    self.assertTrue(finding['status'])
                    self.assertTrue(finding['record_id'])

    def test_each_fixture_demonstrates_at_least_one_hold_or_review(self):
        for code in IMPLEMENTATIONS:
            with self.subTest(workflow=code):
                statuses = [x['status'] for x in run_workflow(code, self.fixture(code), AS_OF)]
                self.assertTrue(any(x.startswith(('HOLD', 'ESCALATE', 'REVIEW', 'TRIAGE', 'DRAFT', 'SECURITY')) for x in statuses))

    def test_missing_primary_record_fails_closed(self):
        rows = [r for r in self.fixture('AA01') if r['record_type'] == 'EVIDENCE']
        with self.assertRaises(InputError):
            run_workflow('AA01', rows, AS_OF)

    def test_aa01_approved_evidence_requires_unique_control(self):
        rows = self.fixture('AA01')
        first = run_workflow('AA01', rows, AS_OF)[0]
        self.assertEqual(first['status'], 'DRAFT_EVIDENCE_AVAILABLE')
        duplicate = copy.copy(rows[0]); duplicate['id'] = 'E2'; duplicate['_source_row'] = 99
        rows.append(duplicate)
        self.assertEqual(run_workflow('AA01', rows, AS_OF)[0]['status'], 'HOLD_NO_UNIQUE_APPROVED_EVIDENCE')

    def test_aa04_exact_three_way_agreement_and_mismatch(self):
        results = run_workflow('AA04', self.fixture('AA04'), AS_OF)
        self.assertEqual([x['status'] for x in results], ['REVIEW_THREE_WAY_AGREEMENT', 'HOLD_THREE_WAY_EXCEPTION'])

    def test_aa04_invalid_zero_or_negative_values_do_not_pass(self):
        for value in ('0', '-10.00'):
            with self.subTest(value=value):
                rows = self.fixture('AA04')
                rows[0]['quantity'] = value
                findings = run_workflow('AA04', rows, AS_OF)
                self.assertEqual(findings[0]['status'], 'HOLD_THREE_WAY_EXCEPTION')

    def test_aa04_empty_supplier_or_currency_do_not_pass(self):
        for key in ('supplier', 'currency'):
            with self.subTest(key=key):
                rows = self.fixture('AA04')
                for row in rows[:3]: row[key] = ''
                self.assertEqual(run_workflow('AA04', rows, AS_OF)[0]['status'], 'HOLD_THREE_WAY_EXCEPTION')

    def test_aa05_escalation_is_not_automated_security_response(self):
        results = run_workflow('AA05', self.fixture('AA05'), AS_OF)
        self.assertEqual(results[-1]['status'], 'ESCALATION_REVIEW')
        self.assertTrue(all(not x['machine_action_taken'] for x in results))

    def test_aa08_emergency_requests_escalated_not_dispatched(self):
        statuses = [x['status'] for x in run_workflow('AA08', self.fixture('AA08'), AS_OF)]
        self.assertIn('ESCALATE_EMERGENCY', statuses)
        self.assertIn('REVIEW_CONTRACTOR_CANDIDATES', statuses)

    def test_aa09_optout_is_not_contacted(self):
        findings = run_workflow('AA09', self.fixture('AA09'), AS_OF)
        self.assertEqual(findings[0]['status'], 'REVIEW_RECOVERY_ELIGIBILITY')
        self.assertEqual(findings[1]['status'], 'HOLD_CONTACT_OR_DATA')
        self.assertFalse(any(x['machine_action_taken'] for x in findings))

    def test_aa10_unmatched_service_requires_baseline(self):
        findings = run_workflow('AA10', self.fixture('AA10'), AS_OF)
        self.assertIn('HOLD_BASELINE', [x['status'] for x in findings])

    def test_aa13_missing_identity_hold(self):
        findings = run_workflow('AA13', self.fixture('AA13'), AS_OF)
        self.assertEqual(findings[1]['status'], 'HOLD_IDENTITY_OR_AUTHORITY')

    def test_aa16_expired_credentials_hold(self):
        findings = run_workflow('AA16', self.fixture('AA16'), AS_OF)
        self.assertEqual(findings[1]['status'], 'HOLD_EXPIRY_OR_VERIFICATION')

    def test_aa18_missing_packet_metadata_hold(self):
        findings = run_workflow('AA18', self.fixture('AA18'), AS_OF)
        self.assertEqual(findings[1]['status'], 'HOLD_MISSING_PACKET_ITEMS')

    def test_regulated_and_security_sensitive_workflows_have_synthetic_gates(self):
        self.assertEqual(SENSITIVE_SYNTHETIC_ONLY, {'AA12', 'AA13', 'AA14', 'AA16', 'AA17', 'AA18'})
        with tempfile.TemporaryDirectory() as tmp:
            process = subprocess.run([sys.executable, '-m', 'avamale_suite', 'run', 'AA13',
                '--data', str(ROOT / 'samples' / 'AA13.csv'), '--out', str(Path(tmp) / 'out')],
                cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('synthetic-only', process.stderr)

    def test_cli_writes_reports_without_claiming_certification(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-m', 'avamale_suite', 'run', 'AA01', '--synthetic',
                '--data', str(ROOT / 'samples' / 'AA01.csv'), '--out', str(Path(tmp) / 'aa01')],
                cwd=ROOT, text=True, capture_output=True, check=True)
            payload = json.loads(result.stdout)
            self.assertFalse(payload['production_certified'])
            manifest = json.loads((Path(tmp) / 'aa01' / 'manifest.json').read_text())
            self.assertTrue(manifest['human_review_required'])
            self.assertTrue(manifest['synthetic_only_demo'])


class ConsentAndFeedbackTests(unittest.TestCase):
    def test_optin_dedupe_triage_and_deletion(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'private' / 'feedback.jsonl'
            kwargs = dict(path=path, code='AA02', category='feature_request',
                description='The current exception list needs an exported reviewer column.',
                impact='high', source_type='synthetic_test', consent=True, source_hint='test_demo_01')
            record = add_feedback(**kwargs)
            self.assertTrue(record['stored'])
            self.assertFalse(add_feedback(**kwargs)['stored'])
            self.assertEqual(len(summarize(path)), 1)
            self.assertEqual(summarize(path)[0]['triage_score'], 3)
            self.assertTrue(delete_feedback(path, record['id']))
            self.assertEqual(summarize(path), [])
            self.assertFalse(delete_feedback(path, record['id']))

    def test_feedback_goes_to_planned_and_shipped_only_after_human_review(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'feedback.jsonl'
            record = add_feedback(path, 'AA02', 'bug',
                'Matching review should show an explicit reason for ambiguous rows.',
                'high', 'synthetic_test', True)
            with self.assertRaises(InputError):
                set_feedback_status(path, record['id'], 'triaged', reviewed=False)
            with self.assertRaises(InputError):
                set_feedback_status(path, record['id'], 'shipped', reviewed=True)
            for state in ('triaged', 'planned', 'shipped'):
                self.assertEqual(set_feedback_status(path, record['id'], state, reviewed=True)['status'], state)
            self.assertEqual(summarize(path)[0]['issues'][0]['status'], 'shipped')

    def test_reject_unconsented_and_identifying_comments(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'feedback.jsonl'
            base = ['AA02', 'bug', 'The row matching logic sometimes misses matching rows.',
                    'medium', 'consented_interview']
            with self.assertRaises(InputError): add_feedback(path, *base, consent=False)
            self.assertFalse(path.exists())
            for identifying in ['Contact accountant@example.com to discuss data.',
                                'The app reported account 123456789012 to the operator.',
                                'Use https://example.com/source to access the client file.']:
                with self.subTest(text=identifying):
                    with self.assertRaises(InputError):
                        add_feedback(path, 'AA02', 'bug', identifying, 'medium', 'consented_interview', True)
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
