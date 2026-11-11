"""Product definitions taken from the AVAMALE AA01-AA18 offer register.

These product surfaces support self-contained, human-reviewed case work. They do
not represent vendor integrations, industry certifications or 18 hosted SaaS apps.
"""
from __future__ import annotations

from avamale_suite.workflows import SPEC, SENSITIVE_SYNTHETIC_ONLY

# Each tuple: buyer, practical job, human deliverable, disclosure, queue label.
DESCRIPTIONS = {
    'AA01': ('Security and sales engineering', 'Collect approved questionnaire evidence against control IDs', 'Reviewer-controlled response evidence map', 'Evidence provenance and current approval require human confirmation', 'Evidence requests'),
    'AA02': ('Finance teams and accounting practices', 'Compare structured bank and ledger lines and isolate exceptions', 'Source-row-linked match and exception workbook', 'Does not post journal entries, prove balances or integrate with an ERP', 'Reconciliation runs'),
    'AA03': ('Freight operations and import teams', 'Compare transcribed invoice, packing list and bill-of-lading fields', 'Shipment and item discrepancy list', 'Does not read PDFs, authenticate documents or determine customs clearance', 'Shipment checks'),
    'AA04': ('Accounts payable', 'Compare purchase order, invoice and goods-received records', 'Three-way discrepancy and approval-preparation report', 'No payment posting or purchasing approval', 'Invoice reviews'),
    'AA05': ('IT service desks', 'Spot probable ticket categories and escalate keywords', 'Human routing and escalation review queue', 'No guaranteed priority, ticketing-platform integration or auto-escalation', 'Triage queue'),
    'AA06': ('Payment operations', 'Audit dispute and evidence references before manual submission', 'Chargeback evidence-readiness checklist', 'No payment network submission or promise of won disputes', 'Dispute cases'),
    'AA07': ('Real-estate asset teams', 'Review supplied lease dates, clause excerpts and source references', 'Renewal/notice review checklist', 'No PDF extraction or legal interpretation', 'Lease reviews'),
    'AA08': ('Property operations', 'Find proposed contractor candidates from verified metadata', 'Human-reviewed maintenance assignment suggestions', 'No live availability or autonomous contractor dispatch', 'Maintenance requests'),
    'AA09': ('Recurring-revenue operations', 'Check failed-payment metadata against explicitly recorded contact consent', 'Review-only payment recovery candidate list', 'No actual billing confirmation or automatic customer messaging', 'Recovery queue'),
    'AA10': ('FinOps and engineering', 'Compare cloud billing baseline and actual amounts', 'Explainable anomaly-review items', 'No live cloud account access or promised cost savings', 'Cost reviews'),
    'AA11': ('Procurement and ESG', 'Check required supplier evidence identifiers and expiry dates', 'Supplier evidence-gap list', 'No assurance or independent supplier attestation', 'Supplier reviews'),
    'AA12': ('Regulated compliance teams', 'Prepare alert-to-evidence relationships for specialist review', 'Human AML evidence checklist (synthetic only)', 'Never determines suspicious activity or files regulatory reports', 'Alert cases'),
    'AA13': ('Data protection officers', 'Organize request metadata, authority and deadline checks', 'Human privacy-request preparation queue (synthetic only)', 'Does not verify identity, retrieve data or decide legal deadlines', 'Privacy cases'),
    'AA14': ('Security operations', 'Flag message metadata that requires security analysis', 'Analyst-only phishing incident queue (synthetic only)', 'Does not click links, scan malware or block threats', 'Security queue'),
    'AA15': ('HR and IT operations', 'Cross-check offboarding resources and approver evidence', 'Human offboarding action checklist', 'Does not disable access, reclaim licences or verify identities', 'Offboarding cases'),
    'AA16': ('Healthcare credentialing teams', 'Check provided credential expiry/source metadata', 'Credential evidence exception list (synthetic only)', 'Does not perform primary-source verification or authorize clinicians', 'Credential reviews'),
    'AA17': ('Healthcare intake operations', 'Check declared intake field and source completeness', 'Manual intake corrections queue (synthetic only)', 'Does not interact with EHR or process patient health records', 'Intake cases'),
    'AA18': ('Revenue cycle and medical teams', 'Check prior-authorization evidence metadata types', 'Human packet-preparation checklist (synthetic only)', 'No clinical appropriateness or payer coverage determination', 'Packet cases'),
}


# Reviewed product hypotheses: no customer ROI has been independently verified.
FLAGSHIP_PROOFS = {
  'AA03': {'pain': 'Invoices, packing lists and bills of lading may disagree; staff recheck or amend shipments by hand.',
           'approach': 'Compare verified, transcribed document rows and highlight conflicts with source references.',
           'proof': 'Measure mismatch frequency, end-to-end review time including transcription, missed discrepancies and false alarms.',
           'buyer_question': 'Which document mismatch last forced your team to recheck or amend a shipment?',
           'priority_label': 'Lead discovery lane'},
  'AA01': {'pain': 'Repeated enterprise security questionnaires consume security and sales-engineering time.',
           'approach': 'Map questions to approved control evidence and hold unsupported answers for reviewer approval.',
           'proof': 'Measure full reviewer time for 20 questions and traceable, approved evidence coverage.',
           'buyer_question': 'Which buyer security questions still consume time despite approved evidence?',
           'priority_label': 'Evidence validation lane'},
  'AA04': {'pain': 'Purchase-order, invoice and goods-received mismatches become manual accounts-payable exceptions.',
           'approach': 'Create a human-controlled three-way mismatch report from structured source records.',
           'proof': 'Measure exception handling minutes, missed mismatches, false positives and reviewer overrides.',
           'buyer_question': 'Which invoice exceptions remain unresolved after your existing ERP checks?',
           'priority_label': 'Procurement discovery lane'},
  'AA02': {'pain': 'Existing accounting systems may leave a small, costly tail of ambiguous bank-ledger exceptions.',
           'approach': 'Identify deterministic structured matches and hold duplicates or ambiguous lines for human review.',
           'proof': 'First verify recurring residual exceptions exist; then compare complete reviewer time and false-match rates.',
           'buyer_question': 'Which reconciliation exception cannot your current accounting package resolve?',
           'priority_label': 'Conditional: confirm unmet need'}
}
FLAGSHIP_ORDER = ('AA03', 'AA01', 'AA04', 'AA02')

SECTORS = {
    'AA01': 'B2B SaaS', 'AA02': 'Accounting', 'AA03': 'Logistics', 'AA04': 'Procurement',
    'AA05': 'IT support', 'AA06': 'E-commerce', 'AA07': 'Real estate', 'AA08': 'Property',
    'AA09': 'Subscriptions', 'AA10': 'Cloud finance', 'AA11': 'Supply chain',
    'AA12': 'Financial compliance', 'AA13': 'Privacy', 'AA14': 'Cybersecurity',
    'AA15': 'People operations', 'AA16': 'Healthcare', 'AA17': 'Healthcare',
    'AA18': 'Healthcare',
}


def catalog() -> list[dict]:
    result = []
    for code, (title, schema) in sorted(SPEC.items()):
        buyer, job, output, caveat, queue = DESCRIPTIONS[code]
        inputs = ('bank', 'ledger') if code == 'AA02' else ('data',)
        columns = ['id,date,amount,currency,reference'] * 2 if code == 'AA02' else (
            ['id,shipment_id,document_type,document_ref,item_code,quantity,packages,gross_weight_kg,consignee,origin_country']
            if code == 'AA03' else [schema]
        )
        fit = FLAGSHIP_PROOFS.get(code, {})
        result.append({
            'featured': code in FLAGSHIP_PROOFS, 'priority_label': fit.get('priority_label', 'Extended synthetic workflow'),
            'pain': fit.get('pain', ''), 'approach': fit.get('approach', ''),
            'proof': fit.get('proof', ''), 'buyer_question': fit.get('buyer_question', ''),
            'id': code, 'name': title, 'sector': SECTORS[code], 'buyer': buyer,
            'job': job, 'deliverable': output, 'limitations': caveat,
            'queue': queue, 'inputs': list(inputs), 'columns': columns,
            'sensitive': code in SENSITIVE_SYNTHETIC_ONLY,
            'demo_only': True, 'human_review_required': True,
            'production_certified': False,
        })
    return result


PRODUCTS = {p['id']: p for p in catalog()}
assert set(PRODUCTS) == {f'AA{i:02}' for i in range(1, 19)}, '18 product definitions are required'
