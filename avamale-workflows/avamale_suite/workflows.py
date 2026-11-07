"""Rule-bound, local, non-destructive review adapters for AA01 and AA04--AA18.

No external API, no automated decisions, no regulatory or customer outcome claims.
Every output is a suggested review item, not an approval or completed action.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation
import re

from avamale_review import InputError, norm

SPEC = {
 'AA01': ('Security questionnaire evidence preparation', 'id,record_type,control_id,content,approved,source_ref'),
 'AA02': ('Bank statement to ledger exception review', 'See AA02 bank+ledger schemas'),
 'AA03': ('Freight document discrepancy review', 'See AA03 item-level schema'),
 'AA04': ('Accounts payable three-way matching', 'id,record_type,po_id,item_code,supplier,quantity,unit_price,currency'),
 'AA05': ('IT helpdesk request triage', 'id,ticket_ref,summary,reported_priority,category'),
 'AA06': ('Chargeback evidence packet preparation', 'id,record_type,dispute_ref,transaction_ref,amount,currency,evidence_ref'),
 'AA07': ('Lease date and clause evidence review', 'id,lease_ref,effective_on,ends_on,notice_due,clause_excerpt,source_ref'),
 'AA08': ('Maintenance dispatch suggestions', 'id,record_type,request_ref,region,trade,urgency,available,verified'),
 'AA09': ('Failed subscription payment review', 'id,invoice_ref,customer_ref,amount,currency,days_overdue,consent,unsubscribed'),
 'AA10': ('Cloud cost anomaly review', 'id,record_type,service,period,cost,currency'),
 'AA11': ('Supplier ESG evidence collection', 'id,record_type,supplier_ref,requirement,evidence_ref,verified,expires_on'),
 'AA12': ('KYC/AML alert evidence triage', 'id,record_type,alert_ref,reason_code,evidence_ref,verified'),
 'AA13': ('Data subject access request case preparation', 'id,case_ref,request_type,identity_verified,received_on,deadline,authority_ref'),
 'AA14': ('Phishing alert metadata triage', 'id,message_ref,authentication_result,suspicious_urls,attachments,evidence_ref'),
 'AA15': ('Employee offboarding checklist review', 'id,worker_ref,resource_ref,action_type,completion_verified,approver_ref'),
 'AA16': ('Healthcare credential evidence review', 'id,provider_ref,credential_type,expires_on,source_ref,verified'),
 'AA17': ('EHR intake field completeness review', 'id,case_ref,field_name,provided,source_ref'),
 'AA18': ('Medical prior-authorisation packet review', 'id,record_type,case_ref,evidence_type,evidence_ref,verified'),
}

# Every rule-bound adapter must receive nonempty source linking keys.
# Missing keys are a validation error, not a reason to claim a tentative match.
LINK_FIELDS = {
    'AA01': ('control_id',), 'AA04': ('po_id', 'item_code'),
    'AA05': ('ticket_ref',), 'AA06': ('dispute_ref', 'transaction_ref'),
    'AA07': ('lease_ref',), 'AA08': ('region', 'trade'),
    'AA09': ('invoice_ref', 'customer_ref'), 'AA10': ('service', 'currency', 'period'),
    'AA11': ('supplier_ref', 'requirement'), 'AA12': ('alert_ref',),
    'AA13': ('case_ref', 'request_type'), 'AA14': ('message_ref',),
    'AA15': ('worker_ref', 'resource_ref', 'action_type'),
    'AA16': ('provider_ref', 'credential_type'),
    'AA17': ('case_ref', 'field_name'), 'AA18': ('case_ref',),
}
RECORD_TYPES = {
    'AA01': {'EVIDENCE', 'QUESTION'},
    'AA04': {'PO', 'INVOICE', 'RECEIPT'},
    'AA06': {'DISPUTE', 'EVIDENCE'},
    'AA08': {'CONTRACTOR', 'REQUEST'},
    'AA10': {'BASELINE', 'ACTUAL'},
    'AA11': {'REQUIREMENT', 'EVIDENCE'},
    'AA12': {'ALERT', 'EVIDENCE'},
    'AA18': {'CASE', 'EVIDENCE'},
}

# Operational and regulated workflows are intentionally synthetic/test-only.
SENSITIVE_SYNTHETIC_ONLY = {'AA12', 'AA13', 'AA14', 'AA16', 'AA17', 'AA18'}


def finding(row, status, check, description, next_action, *, related=None, confidence='RULE_ONLY'):
    return {
        'status': status,
        'record_id': row.get('id', ''),
        'source_rows': str(row.get('_source_row', '')),
        'related_record_ids': ' | '.join(str(x) for x in (related or [])),
        'check': check,
        'finding': description,
        'recommended_human_action': next_action,
        'confidence_basis': confidence,
        'review_required': True,
        'machine_action_taken': False,
    }


def related_group(rows, *fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(norm(row.get(f, '')) for f in fields)].append(row)
    return groups


def kind(row):
    return norm(row.get('record_type'))


def affirmative(value):
    return norm(value) in {'YES', 'TRUE', '1', 'Y'}


def dec(value):
    try:
        text = str(value).strip()
        if not re.fullmatch(r'[+-]?\d+(?:\.\d{1,2})?', text):
            return None
        n = Decimal(text)
        return n if n.is_finite() else None
    except (InvalidOperation, TypeError):
        return None


def iso(value):
    try:
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(value)):
            return None
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def nonnegative_int(value):
    return int(value) if re.fullmatch(r'\d+', str(value).strip()) else None


def aa01(rows, as_of):
    approved = related_group([r for r in rows if kind(r) == 'EVIDENCE' and affirmative(r['approved']) and r['source_ref'].strip()], 'control_id')
    output = []
    for q in [r for r in rows if kind(r) == 'QUESTION']:
        pool = approved.get((norm(q['control_id']),), []) if norm(q['control_id']) else []
        if len(pool) == 1:
            e = pool[0]
            output.append(finding(q,'DRAFT_EVIDENCE_AVAILABLE','Approved control ID match',
                f"An explicitly marked approved evidence excerpt is available; evidence ID {e['id']}, source {e['source_ref']}; approval and currency not independently verified.",
                'Review actual approved source, context and currency before drafting a truthful response.',related=[e['id']]))
        else:
            output.append(finding(q,'HOLD_NO_UNIQUE_APPROVED_EVIDENCE','Approved control ID match',
                f'{len(pool)} approved evidence records found for the control ID.',
                'Find authoritative approval and evidence, then have security owner review.',related=[p['id'] for p in pool]))
    return output


def aa04(rows, as_of):
    output=[]
    for key,group in sorted(related_group(rows,'po_id','item_code').items()):
        first=group[0]; variants=Counter(kind(r) for r in group)
        if not key[0] or not key[1] or any(variants[t] != 1 for t in ('PO','INVOICE','RECEIPT')) or len(group)!=3:
            output.append(finding(first,'HOLD_INCOMPLETE_OR_AMBIGUOUS','Three-way presence',
                f'PO/invoice/receipt occurrences: {dict(variants)}; unique full set required.',
                'Resolve missing, duplicated or amended source documents manually.',related=[r['id'] for r in group[1:]]));continue
        po, inv, receipt = (next(r for r in group if kind(r)==t) for t in ('PO','INVOICE','RECEIPT'))
        valid=all(dec(r['quantity']) is not None and dec(r['quantity']) > 0 and dec(r['unit_price']) is not None and dec(r['unit_price']) > 0 for r in group)
        same_supplier=all(norm(r['supplier']) for r in group) and len({norm(r['supplier']) for r in group})==1
        same_currency=all(re.fullmatch(r'[A-Z]{3}', norm(r['currency'])) for r in group) and len({norm(r['currency']) for r in group})==1
        quantity_ok=valid and dec(inv['quantity']) == dec(receipt['quantity']) == dec(po['quantity'])
        price_ok=valid and dec(inv['unit_price']) == dec(po['unit_price'])
        passed=valid and same_supplier and same_currency and quantity_ok and price_ok
        output.append(finding(first, 'REVIEW_THREE_WAY_AGREEMENT' if passed else 'HOLD_THREE_WAY_EXCEPTION',
            'Exact supplier/currency/quantity/price, unique PO item',
            f'supplier={same_supplier}; currency={same_currency}; quantities_equal={quantity_ok}; invoice_po_unit_price_equal={price_ok}; numerics_valid={valid}.',
            'Finance/AP reviewer must confirm receipts, partial deliveries, tax and approval before payment.',related=[inv['id'],receipt['id']]))
    return output


def aa05(rows, as_of):
    output=[]
    for r in rows:
        combined=norm(r['summary']+' '+r['category'])
        urgent=bool(re.search(r'\b(BREACH|PHISHING|COMPROMISED|OUTAGE|RANSOMWARE|DATA LOSS)\b',combined))
        output.append(finding(r,'ESCALATION_REVIEW' if urgent else 'TRIAGE_SUGGESTION',
            'Keyword-based routing (not threat detection)',
            f"Reported priority {r['reported_priority']}; category {r['category']}; escalation keywords present: {urgent}.",
            'Service desk owner validates urgency, access permissions and assignment; do not auto-close, isolate or alert.'))
    return output


def aa06(rows, as_of):
    output=[]
    for key, group in sorted(related_group(rows,'dispute_ref').items()):
        first=group[0]; disputes=[r for r in group if kind(r)=='DISPUTE']; evidence=[r for r in group if kind(r)=='EVIDENCE' and r['evidence_ref'].strip()]
        valid=(len(disputes)==1 and bool(evidence) and dec(disputes[0]['amount']) is not None and dec(disputes[0]['amount']) > 0 and bool(re.fullmatch(r'[A-Z]{3}', norm(disputes[0]['currency']))))
        if valid:
            dispute=disputes[0]
            valid=all(norm(e['transaction_ref'])==norm(dispute['transaction_ref'])
                and norm(e['currency'])==norm(dispute['currency'])
                and dec(e['amount'])==dec(dispute['amount']) for e in evidence)
        output.append(finding(first,'REVIEW_PACKET_METADATA' if valid else 'HOLD_INCOMPLETE_DISPUTE',
            'Dispute and evidence metadata presence',
            f'{len(disputes)} dispute record(s); {len(evidence)} linked evidence reference(s); no evidence content verified.',
            'Payments owner must verify consent, deadlines, network rules and source documents before submission.',related=[r['id'] for r in group[1:]]))
    return output


def aa07(rows, as_of):
    output=[]
    for r in rows:
        start,end,notice=iso(r['effective_on']),iso(r['ends_on']),iso(r['notice_due'])
        ok=bool(start and end and notice and start <= notice <= end and r['clause_excerpt'].strip() and r['source_ref'].strip())
        output.append(finding(r,'REVIEW_LEASE_EVIDENCE' if ok else 'HOLD_LEASE_DATE_OR_SOURCE',
            'Pre-extracted dates, clause and source reference',
            f'valid_date_sequence={bool(start and end and notice and start<=notice<=end)}; source_and_excerpt_present={bool(r["clause_excerpt"].strip() and r["source_ref"].strip())}.',
            'Legal/property reviewer verifies extraction, amendments, notice interpretation and jurisdiction.'))
    return output


def aa08(rows, as_of):
    contractors=[r for r in rows if kind(r)=='CONTRACTOR' and affirmative(r['available']) and affirmative(r['verified'])]
    output=[]
    for req in [r for r in rows if kind(r)=='REQUEST']:
        urgent=norm(req['urgency']) in {'EMERGENCY','LIFE_SAFETY','CRITICAL'}
        eligible=[x for x in contractors if norm(x['region'])==norm(req['region']) and norm(x['trade'])==norm(req['trade'])]
        known_urgency={'LOW','NORMAL','MEDIUM','HIGH','EMERGENCY','LIFE_SAFETY','CRITICAL'}
        status=('HOLD_INVALID_URGENCY' if norm(req['urgency']) not in known_urgency else
            'ESCALATE_EMERGENCY' if urgent else
            'REVIEW_CONTRACTOR_CANDIDATES' if eligible else 'HOLD_NO_VERIFIED_CONTRACTOR')
        output.append(finding(req,status,'Location/trade/availability/verification metadata',
            f'{len(eligible)} possible contractor(s); emergency escalation={urgent}.',
            'Property manager must confirm safety, current availability, contract, price and dispatch.',related=[r['id'] for r in eligible]))
    return output


def aa09(rows, as_of):
    output=[]
    for r in rows:
        amount,days=dec(r['amount']),nonnegative_int(r['days_overdue'])
        explicit_no=norm(r['unsubscribed']) in {'NO','FALSE','0','N'}
        eligible=bool(amount is not None and amount > 0 and days is not None and days>0 and affirmative(r['consent']) and explicit_no)
        output.append(finding(r,'REVIEW_RECOVERY_ELIGIBILITY' if eligible else 'HOLD_CONTACT_OR_DATA',
            'Failed payment, consent and opt-out check',
            f'amount_valid={amount is not None}; days_overdue={days}; consent={affirmative(r["consent"])}; opt_out={affirmative(r["unsubscribed"])}.',
            'Revenue operations checks payment state, contact preferences and lawful messaging before sending; no message sent.'))
    return output


def aa10(rows, as_of):
    output=[]
    bases=related_group([r for r in rows if kind(r)=='BASELINE'],'service','currency')
    for r in [r for r in rows if kind(r)=='ACTUAL']:
        baseline=bases.get((norm(r['service']),norm(r['currency'])),[])
        amount=dec(r['cost']); ref=dec(baseline[0]['cost']) if len(baseline)==1 else None
        good=bool(amount is not None and amount >= 0 and ref is not None and ref > 0)
        change=(amount-ref)/ref if good else None
        status=('REVIEW_ANOMALY' if change is not None and change>=Decimal('0.25') and amount-ref>=Decimal('100') else 'REVIEW_NORMAL_RANGE') if good else 'HOLD_BASELINE'
        output.append(finding(r,status,'Illustrative >=25% and >=100 absolute currency-unit increase',
            f'cost={amount}; baseline={ref}; ratio={str(change) if change is not None else "unknown"}; thresholds are demo assumptions.',
            'FinOps owner checks usage, credits, tax, seasonality and negotiated thresholds; do not terminate resources.', related=[b['id'] for b in baseline]))
    return output


def aa11(rows, as_of):
    evidence=related_group([r for r in rows if kind(r)=='EVIDENCE'],'supplier_ref','requirement')
    output=[]
    for req in [r for r in rows if kind(r)=='REQUIREMENT']:
        docs=evidence.get((norm(req['supplier_ref']),norm(req['requirement'])),[])
        valid=[r for r in docs if r['evidence_ref'].strip() and affirmative(r['verified']) and iso(r['expires_on']) is not None and iso(r['expires_on'])>=as_of]
        output.append(finding(req,'REVIEW_EVIDENCE_PRESENT' if len(valid)==1 and len(docs)==1 else 'HOLD_MISSING_OR_UNVERIFIED_EVIDENCE',
            'Required supplier evidence metadata and dates',
            f'{len(docs)} candidate(s), {len(valid)} current marked verified; evidence contents not audited.',
            'Procurement/ESG owner validates issuer and claim; no supplier compliance determination.',related=[d['id'] for d in docs]))
    return output


def aa12(rows, as_of):
    evidence=related_group([r for r in rows if kind(r)=='EVIDENCE' and r['evidence_ref'].strip() and affirmative(r['verified'])],'alert_ref')
    output=[]
    for r in [x for x in rows if kind(x)=='ALERT']:
        docs=evidence.get((norm(r['alert_ref']),),[])
        output.append(finding(r,'REVIEW_ALERT_EVIDENCE_METADATA' if docs else 'HOLD_MISSING_EVIDENCE',
            'Evidence metadata presence (no risk decision)',
            f'{len(docs)} evidence reference(s) marked verified; neither identity nor evidence contents independently verified.',
            'Authorized compliance officer handles risk decisions, suspicious activity reporting and retention.', related=[d['id'] for d in docs]))
    return output


def aa13(rows, as_of):
    output=[]
    for r in rows:
        received,deadline=iso(r['received_on']),iso(r['deadline'])
        complete=bool(affirmative(r['identity_verified']) and received and deadline and received<=deadline and r['authority_ref'].strip())
        output.append(finding(r,'REVIEW_DSAR_CASE_METADATA' if complete else 'HOLD_IDENTITY_OR_AUTHORITY',
            'Identity/authority attestation and dates',
            f'identity_marked_verified={affirmative(r["identity_verified"])}; dates_valid={bool(received and deadline and received<=deadline)}; authority_ref_present={bool(r["authority_ref"].strip())}.',
            'Privacy owner confirms legal deadlines, identity and proportionality; no data is retrieved or disclosed.'))
    return output


def aa14(rows, as_of):
    output=[]
    for r in rows:
        urls,attachments=nonnegative_int(r['suspicious_urls']),nonnegative_int(r['attachments'])
        check=norm(r['authentication_result'])
        safe_auth={'PASS','SPF_PASS','DKIM_PASS','DMARC_PASS'}
        flag=urls is None or attachments is None or urls>0 or attachments>0 or check not in safe_auth
        output.append(finding(r,'SECURITY_ANALYST_REVIEW' if flag else 'REVIEW_REPORTED_MESSAGE',
            'Submitted metadata only; no URL fetching or malware analysis',
            f'auth={check}; suspicious_URL_count={urls}; attachment_count={attachments}; evidence_ref_present={bool(r["evidence_ref"].strip())}.',
            'Analyst examines in approved isolated environment; no automated blocking, deletion or clicking.'))
    return output


def aa15(rows, as_of):
    output=[]
    for r in rows:
        complete=affirmative(r['completion_verified']) and bool(r['approver_ref'].strip())
        output.append(finding(r,'REVIEW_OFFBOARDING_EVIDENCE' if complete else 'HOLD_UNVERIFIED_TASK',
            'Checklist completion and approval metadata',
            f'action={r["action_type"]}; marked_complete={affirmative(r["completion_verified"])}; approver_ref_present={bool(r["approver_ref"].strip())}.',
            'HR and IT must verify all relevant systems, legal holds, equipment and approvals; no access changed.'))
    return output


def aa16(rows, as_of):
    output=[]
    for r in rows:
        expiry=iso(r['expires_on'])
        good=bool(expiry and expiry>=as_of and affirmative(r['verified']) and r['source_ref'].strip())
        output.append(finding(r,'REVIEW_CREDENTIAL_EVIDENCE' if good else 'HOLD_EXPIRY_OR_VERIFICATION',
            'Credential date and verification metadata',
            f'expiry={r["expires_on"]}; not_expired_as_of_test_date={bool(expiry and expiry>=as_of)}; marked_verified={affirmative(r["verified"])}.',
            'Credentialing officer independently verifies issuer, scope, sanctions and primary source; no credential approved.'))
    return output


def aa17(rows, as_of):
    output=[]
    for key,group in sorted(related_group(rows,'case_ref').items()):
        first=group[0]; empty=[r['field_name'] for r in group if not affirmative(r['provided']) or not r['source_ref'].strip()]
        duplicates=[k for k,n in Counter(norm(r['field_name']) for r in group).items() if n>1]
        output.append(finding(first,'REVIEW_SYNTHETIC_INTAKE_METADATA' if not empty and not duplicates else 'HOLD_MISSING_INTAKE_FIELDS',
            'Schema presence and source metadata only',
            f'missing_or_unlinked={empty}; duplicate_fields={duplicates}; no patient data content examined.',
            'Clinical records staff review identity, minimum necessary fields and approved mappings; do not update EHR.', related=[r['id'] for r in group[1:]]))
    return output


def aa18(rows, as_of):
    output=[]
    evidence=related_group([r for r in rows if kind(r)=='EVIDENCE'],'case_ref')
    required={'CLINICIAN_NOTE','SERVICE_REQUEST','COVERAGE_INFO'}
    for c in [r for r in rows if kind(r)=='CASE']:
        docs=evidence.get((norm(c['case_ref']),),[])
        valid=[x for x in docs if x['evidence_ref'].strip() and affirmative(x['verified']) and norm(x['evidence_type'])]
        present={norm(x['evidence_type']) for x in valid}
        missing=sorted(required-present)
        duplicates=any(n>1 for n in Counter(norm(x['evidence_type']) for x in docs).values())
        invalid= len(valid)!=len(docs)
        output.append(finding(c,'REVIEW_PACKET_METADATA' if not missing and not invalid and not duplicates else 'HOLD_MISSING_PACKET_ITEMS',
            'Illustrative minimum packet metadata list; payer rules not checked',
            f'marked evidence types={sorted(present)}; missing demo types={missing}; invalid_evidence={invalid}; duplicates={duplicates}; no medical necessity or insurer criteria evaluated.',
            'Authorized clinical/payer team confirms current requirements, patient permission and submission; no authorization made.', related=[d['id'] for d in docs]))
    return output

IMPLEMENTATIONS={
 'AA01':aa01,'AA04':aa04,'AA05':aa05,'AA06':aa06,'AA07':aa07,
 'AA08':aa08,'AA09':aa09,'AA10':aa10,'AA11':aa11,'AA12':aa12,
 'AA13':aa13,'AA14':aa14,'AA15':aa15,'AA16':aa16,'AA17':aa17,'AA18':aa18,
}


def run_workflow(code,rows,as_of):
    if code not in IMPLEMENTATIONS:
        raise InputError(f'No generic adapter for {code}; AA02/AA03 use specialized tested engines')
    if not rows:
        raise InputError('No input rows')
    for row in rows:
        for field in LINK_FIELDS[code]:
            if not norm(row.get(field, '')):
                raise InputError(f'{code} record {row.get("id", "unknown")}: missing required linking field {field}')
        if code in RECORD_TYPES and kind(row) not in RECORD_TYPES[code]:
            raise InputError(f'{code} record {row.get("id", "unknown")}: unsupported record_type {row.get("record_type", "")}')
        if code == 'AA08' and kind(row) == 'REQUEST':
            if not norm(row.get('request_ref', '')) or not norm(row.get('urgency', '')):
                raise InputError(f'{code} record {row.get("id", "unknown")}: request_ref and urgency are required')
    result=IMPLEMENTATIONS[code](rows,as_of)
    if not result:
        raise InputError(f'{code}: no reviewable primary records in submitted dataset')
    return result
