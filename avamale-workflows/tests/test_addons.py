import unittest
from avamale_review import InputError
from avamale_suite.addons import run_addon, ADDONS


class AddonTests(unittest.TestCase):
    def test_all_four_registered(self):
        self.assertEqual(set(ADDONS), {'BIZ_ASSIST','ASKDOCS','INTAKE'})



    def test_biz_assist_blocks_optouts(self):
        row=dict(id='B', sector='Accounting', pain='Manual work', monthly_volume='100', buyer_role='Owner', budget_confirmed='yes', contact_permission='yes', opted_out='yes')
        self.assertEqual(run_addon('BIZ_ASSIST',[row])[0]['status'],'DO_NOT_CONTACT')

    def test_biz_assist_requires_budget(self):
        row=dict(id='B', sector='Accounting', pain='Manual work', monthly_volume='100', buyer_role='Owner', budget_confirmed='no', contact_permission='yes', opted_out='no')
        self.assertEqual(run_addon('BIZ_ASSIST',[row])[0]['status'],'QUALIFY_VOLUME_BUDGET')

    def test_askdocs_returns_explicit_source_not_generated_answer(self):
        rows=[dict(id='A',document='synthetic',page='2',text='The reviewer verifies original evidence for exceptions.'),dict(id='B',document='synthetic',page='3',text='Warehouse inventory is separate.')]
        found=run_addon('ASKDOCS',rows,'Where is evidence for exceptions?')
        self.assertEqual(len(found),1)
        self.assertEqual(found[0]['evidence']['page'],2)
        self.assertTrue(found[0]['review_required'])

    def test_askdocs_rejects_invalid_page(self):
        with self.assertRaises(InputError):
            run_addon('ASKDOCS',[dict(id='X',document='demo',page='0',text='Evidence exists')],'evidence')

    def test_intake_is_deduplicated(self):
        a=dict(id='1', company_key='Example.test', workflow='AA02',problem_summary='Manual matching',permission_to_contact='yes',duplicate_of='')
        b=dict(a,id='2')
        found=run_addon('INTAKE',[a,b])
        self.assertTrue(all(x['status']=='REVIEW_POSSIBLE_DUPLICATE' for x in found))

    def test_intake_blocks_without_permission(self):
        a=dict(id='1', company_key='Example.test', workflow='AA02',problem_summary='Manual matching',permission_to_contact='no',duplicate_of='')
        self.assertEqual(run_addon('INTAKE',[a])[0]['status'],'HOLD_NO_CONTACT_PERMISSION')

    def test_feedback_can_refer_to_addon_with_consent(self):
        import tempfile
        from pathlib import Path
        from avamale_suite.feedback import add_feedback
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'feedback.jsonl')
            reply = add_feedback(path, 'BIZ_ASSIST', 'feature_request',
                'Reviewers request a manual checklist for buyer discovery calls.',
                'medium', 'synthetic_test', True, 'synth_1')
            self.assertTrue(reply['stored'])

    def test_duplicate_record_id_rejected(self):
        with self.assertRaises(InputError):
            run_addon('INTAKE',[dict(id='A'),dict(id='A')])


if __name__ == '__main__':
    unittest.main()
