"""Local, synthetic-only add-on demonstrations for AVAMALE product hypotheses.

These are deterministic decision-support prototypes: they do not contact users,
move money, block payments, issue legal/medical judgements, or use real data.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import re

from avamale_review import InputError

ADDONS = {
    'BIZ_ASSIST': ('Biz Assist buyer-fit and enquiry preparation', 'id,sector,pain,monthly_volume,buyer_role,budget_confirmed,contact_permission,opted_out'),
    'ASKDOCS': ('AskDocs evidence-linked passage retrieval', 'id,document,page,text'),
    'INTAKE': ('Avamale enquiry routing and quality review', 'id,company_key,workflow,problem_summary,permission_to_contact,duplicate_of'),
}


def _bool(value: str) -> bool:
    return str(value).strip().casefold() in {'yes', 'y', 'true', '1'}


def _nonnegative(value: str, *, integer: bool = False):
    try:
        n = Decimal(str(value).strip())
        if not n.is_finite() or n < 0 or (integer and n != n.to_integral_value()):
            raise ValueError('not nonnegative')
        return int(n) if integer else n
    except (InvalidOperation, ValueError):
        raise InputError('Invalid nonnegative numerical input')


def _risk_bool(value: str, field: str) -> bool:
    """Fail closed on unknown risk signals rather than interpreting them as safe."""
    text = str(value).strip().casefold()
    if text in {'yes', 'y', 'true', '1'}:
        return True
    if text in {'no', 'n', 'false', '0'}:
        return False
    raise InputError(f'{field}: use yes/no; unknown signals require human review, not a low-risk score')


def _result(row, status, evidence, next_action, **extra):
    return {
        'record_id': row['id'], 'status': status, 'evidence': evidence,
        'next_human_action': next_action, 'review_required': True,
        'autonomous_action_taken': False, **extra,
    }




def biz_assist(records):
    output = []
    for row in records:
        monthly = _nonnegative(row['monthly_volume'], integer=True)
        permitted = _bool(row['contact_permission']) and str(row['opted_out']).strip().casefold() in {'no', 'n', 'false', '0'}
        demand = bool(row['pain'].strip())
        purchaser = bool(row['buyer_role'].strip())
        budget = _bool(row['budget_confirmed'])
        if not permitted:
            status = 'DO_NOT_CONTACT'
        elif not (demand and purchaser):
            status = 'NEEDS_DISCOVERY'
        elif monthly >= 20 and budget:
            status = 'PRIORITIZE_MANUAL_DISCOVERY'
        else:
            status = 'QUALIFY_VOLUME_BUDGET'
        output.append(_result(row, status, {
            'monthly_volume': monthly, 'problem_described': demand,
            'buyer_role_present': purchaser, 'budget_self_reported': budget,
            'contact_permission': permitted},
            'Sales owner verifies need, role, scope and lawful outreach. Never send automatically.'))
    return output


def _tokens(text):
    stop = {'the', 'and', 'for', 'from', 'with', 'that', 'this', 'what', 'where', 'when', 'are', 'how', 'can', 'will', 'is', 'a', 'an', 'of', 'to', 'in', 'on'}
    return {t for t in re.findall(r'[a-z0-9]{2,}', text.casefold()) if t not in stop}


def askdocs(records, question):
    """Return snippets/source references, not generated answers or true citations to originals."""
    keywords = _tokens(question)
    if not keywords:
        raise InputError('Use a meaningful question with searchable terms')
    matches = []
    for row in records:
        try:
            page = int(row['page'])
        except (ValueError, TypeError):
            raise InputError('Page must be a positive integer')
        if page < 1:
            raise InputError('Page must be a positive integer')
        tokens = _tokens(row['text'])
        overlap = len(keywords.intersection(tokens))
        if overlap:
            matches.append(_result(row, 'SOURCE_PASSAGE_CANDIDATE', {
                'document': row['document'], 'page': page, 'matched_query_terms': overlap,
                'excerpt': row['text'][:500]},
                'Reader confirms the quoted passage in its original file before using it.',
                retrieval_score=overlap / len(keywords), basis='LEXICAL_DEMO_RETRIEVAL'))
    matches.sort(key=lambda m: (-m['retrieval_score'], m['record_id']))
    return matches[:5]


def intake(records):
    observed = Counter(row['company_key'].casefold().strip() for row in records if row['company_key'].strip())
    out = []
    for row in records:
        valid = bool(row['company_key'].strip() and row['problem_summary'].strip() and row['workflow'].strip())
        duplicate = bool(row['duplicate_of'].strip() or observed[row['company_key'].casefold().strip()] > 1)
        permitted = _bool(row['permission_to_contact'])
        status = ('HOLD_NO_CONTACT_PERMISSION' if not permitted else
                  'HOLD_INCOMPLETE' if not valid else
                  'REVIEW_POSSIBLE_DUPLICATE' if duplicate else 'READY_FOR_HUMAN_QUALIFICATION')
        out.append(_result(row, status, {
            'fields_present': valid, 'contact_permission': permitted,
            'possible_duplicate': duplicate, 'requested_workflow': row['workflow']},
            'Human reviews source, identity, requirements and sender suppression status before response.'))
    return out


def run_addon(name, records, question=''):
    if name not in ADDONS:
        raise InputError('Unknown add-on')
    if len(records) > 5000:
        raise InputError('Demo size limit is 5000 rows')
    ids = [r.get('id', '').strip() for r in records]
    if not records or not all(ids) or len(set(ids)) != len(ids):
        raise InputError('Require at least one row and unique nonempty IDs')
    if name == 'BIZ_ASSIST': return biz_assist(records)
    if name == 'ASKDOCS': return askdocs(records, question)
    return intake(records)
