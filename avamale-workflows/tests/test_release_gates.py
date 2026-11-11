"""Release preflight regression tests: zero network or outbound side effects."""
import unittest
from avamale_suite.release_gates import (
    FORM_ID, preflight_send, normalize_wix_submission, classify_pilot_upload,
)

class ReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.cfg={'PAUSE':'TRUE','ARMED':'FALSE','DRY_RUN':'TRUE'}
        self.contact={'email':'buyer@example.com','verified':'Y','status':''}
        self.audit=dict(reply_audit_complete=True,suppression_complete=True,
                        sender_identity_verified=True,delivery_provider_verified=True)
        self.form={'id':'wix-abc','formId':FORM_ID,'status':'CONFIRMED','submissions':{
            'first_name_avsample':'Jane','email_avsample':'JANE@EXAMPLE.COM',
            'company_avsample':'Example Trading',
            'service_avsample':'Freight document discrepancies',
            'bottleneck_avsample':'Compare invoices'}}

    def send(self, **overrides):
        return preflight_send(config=self.cfg,contact=self.contact,
                              **{**self.audit,**overrides})

    def intake(self, **kwargs):
        return normalize_wix_submission(self.form,
                 known_ids=kwargs.get('known_ids',set()),
                 suppressed_emails=kwargs.get('suppressed_emails',set()))

    def test_paused(self):
        self.assertEqual(self.send().code,'PAUSED')
    def test_unarmed(self):
        self.cfg['PAUSE']='FALSE'
        self.assertEqual(self.send().code,'UNARMED')
    def test_dry_run(self):
        self.cfg.update(PAUSE='FALSE',ARMED='TRUE')
        self.assertEqual(self.send().code,'DRY_RUN')
    def test_missing_reply_audit_fails_closed(self):
        self.cfg.update(PAUSE='FALSE',ARMED='TRUE',DRY_RUN='FALSE')
        self.assertEqual(self.send(reply_audit_complete=False).code,'AUDIT_INCOMPLETE')
    def test_bounce_and_reply_are_terminal(self):
        self.cfg.update(PAUSE='FALSE',ARMED='TRUE',DRY_RUN='FALSE')
        for status in ('BOUNCED','REPLIED_HOLD','DO_NOT_CONTACT','MANUAL_MERGE'):
            with self.subTest(status=status):
                self.contact['status']=status
                self.assertEqual(self.send().code,'TERMINAL')
    def test_optout_blocks(self):
        self.cfg.update(PAUSE='FALSE',ARMED='TRUE',DRY_RUN='FALSE')
        self.contact['opted_out']='TRUE'
        self.assertEqual(self.send().code,'SUPPRESSED')
    def test_authorized_preflight_does_not_send(self):
        self.cfg.update(PAUSE='FALSE',ARMED='TRUE',DRY_RUN='FALSE')
        self.assertEqual(self.send().code,'PRECHECK_ONLY')
    def test_wix_pending_rejected(self):
        self.form['status']='PENDING'
        self.assertEqual(self.intake()[0].code,'NOT_CONFIRMED')
    def test_wix_duplicate_rejected(self):
        self.assertEqual(self.intake(known_ids={'wix-abc'})[0].code,'DUPLICATE_OR_MISSING_ID')
    def test_wix_suppression_rejected(self):
        self.assertEqual(self.intake(suppressed_emails={'jane@example.com'})[0].code,'SUPPRESSED')
    def test_wix_fields_normalized(self):
        status,row=self.intake()
        self.assertTrue(status.allowed)
        self.assertEqual(row['wix_submission_id'],'wix-abc')
        self.assertEqual(row['email'],'jane@example.com')
    def test_wix_internal_qa_dropped(self):
        self.form['submissions']['company_avsample']='AVAMALE INTERNAL QA'
        self.assertEqual(self.intake()[0].code,'QA_EXCLUDED')
    def test_wix_wrong_form_rejected(self):
        self.form['formId']='another-form'
        self.assertEqual(self.intake()[0].code,'NOT_CONFIRMED')
    def test_real_documents_blocked(self):
        self.assertEqual(classify_pilot_upload(filename='bank.csv',
           documented_synthetic=False,safe_intake_approved=False).code,'REAL_DATA_HOLD')
    def test_pdfs_not_accepted_by_structured_engine(self):
        self.assertEqual(classify_pilot_upload(filename='invoice.pdf',
           documented_synthetic=True,safe_intake_approved=False).code,'UNSUPPORTED_FORMAT')
    def test_synthetic_structured_data_still_requires_review(self):
        self.assertEqual(classify_pilot_upload(filename='freight.csv',
           documented_synthetic=True,safe_intake_approved=False).code,'REVIEW_REQUIRED')

if __name__=='__main__':
    unittest.main()
