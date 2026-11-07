"""Consent-gated, minimised, local-only product feedback registry.

This tool deliberately does not ingest raw emails, customer documents or private comments.
Enter an approved, de-identified problem statement; all feedback remains human-triaged.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import re
import tempfile

from avamale_review import InputError
from .workflows import SPEC
from .addons import ADDONS

CATEGORIES={'bug','feature_request','sales_objection','validation','security','usability'}
STATES={'new','triaged','planned','shipped','rejected'}
# This is a conservative screening filter, not a de-identification or DLP guarantee.
SENSITIVE_PATTERNS=[
    re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',re.I),
    re.compile(r'https?://\S+',re.I),
    re.compile(r'\+?\d[\s\-()\d]{9,}\d'),
    re.compile(r'\b\d{10,}\b'),
]


def clean_description(text):
    text=' '.join(str(text).strip().split())
    if not 15<=len(text)<=400:
        raise InputError('Provide a 15-400 character high-level problem/feature statement, not a raw customer quote.')
    if any(p.search(text) for p in SENSITIVE_PATTERNS):
        raise InputError('Possible email, URL, phone or account number found; paraphrase without identifiers.')
    return text


def read_records(path):
    p=Path(path)
    if not p.exists():
        return []
    records=[]
    for i, line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
        if line.strip():
            try: records.append(json.loads(line))
            except json.JSONDecodeError as exc: raise InputError(f'Invalid feedback registry line {i}') from exc
    return records


def add_feedback(path,code,category,description,impact,source_type,consent,source_hint=''):
    if not consent:
        raise InputError('Do not record customer comments without permission/approved internal summarisation.')
    if (code not in SPEC and code not in ADDONS) or category not in CATEGORIES or source_type not in {'consented_interview','approved_deidentified_summary','synthetic_test','support_ticket_optin'}:
        raise InputError('Unsupported workflow/category/source type')
    if impact not in {'low','medium','high'}:
        raise InputError('Impact must be low, medium or high')
    text=clean_description(description)
    hint=' '.join(source_hint.split()).strip()
    if len(hint)>40 or (hint and not re.fullmatch(r'[A-Za-z0-9_-]+',hint)):
        raise InputError('source_hint must be an anonymous opaque code with <=40 alphanumeric/underscore/dash characters')
    digest=hashlib.sha256((code+'|'+category+'|'+text.casefold()).encode()).hexdigest()[:18]
    entries=read_records(path)
    if any(x.get('id')==digest for x in entries):
        return {'id':digest,'stored':False,'reason':'duplicate'}
    entry={
        'id':digest,'workflow':code,'category':category,'summary':text,
        'impact':impact,'source_type':source_type,'source_hint':hint,'status':'new',
        'consent_confirmed_by_operator':True,'observed_at_utc':datetime.now(timezone.utc).isoformat(),
        'personal_data_in_scope':False,'customer_data_uploaded':False,
    }
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    # Local log only. Parent folder ACL and backup policy remain the operator's responsibility.
    with p.open('a',encoding='utf-8') as f:
        f.write(json.dumps(entry,ensure_ascii=False)+'\n')
    try: os.chmod(p,0o600)
    except OSError: pass
    return {'id':digest,'stored':True,'status':'new'}


def summarize(path):
    groups=defaultdict(lambda:{'count':0,'high':0,'medium':0,'items':[]})
    for e in read_records(path):
        if e.get('status')=='rejected': continue
        key=(e['workflow'],e['category'])
        group=groups[key]
        group['count']+=1
        if e['impact'] in {'high','medium'}: group[e['impact']]+=1
        group['items'].append({'id':e['id'],'summary':e['summary'],'status':e['status']})
    output=[]
    for (code,category),data in groups.items():
        # Triage hint only: no automatic roadmap prioritisation or feature releases.
        score=3*data['high']+2*data['medium']+(data['count']-data['high']-data['medium'])
        output.append({'workflow':code,'category':category,'signals':data['count'],'triage_score':score,'issues':data['items']})
    return sorted(output,key=lambda x:(-x['triage_score'],x['workflow'],x['category']))


def delete_feedback(path,record_id):
    p=Path(path);entries=read_records(path);kept=[x for x in entries if x.get('id')!=record_id]
    if len(kept)==len(entries):return False
    with tempfile.NamedTemporaryFile('w',delete=False,dir=p.parent,encoding='utf-8') as tmp:
        for e in kept:tmp.write(json.dumps(e,ensure_ascii=False)+'\n')
        tmp_path=Path(tmp.name)
    os.chmod(tmp_path,0o600)
    os.replace(tmp_path,p)
    return True


def set_feedback_status(path, record_id, new_status, reviewed=False):
    """A reviewer (not an automation) advances a de-identified product observation."""
    if not reviewed:
        raise InputError('A human reviewer must explicitly confirm feedback state changes.')
    if new_status not in STATES:
        raise InputError('Invalid feedback state')
    allowed = {
        'new': {'triaged', 'rejected'},
        'triaged': {'planned', 'rejected'},
        'planned': {'shipped', 'rejected'},
        'shipped': set(),
        'rejected': set(),
    }
    p = Path(path)
    entries = read_records(path)
    found = False
    for entry in entries:
        if entry.get('id') == record_id:
            found = True
            current = entry.get('status', 'new')
            if new_status not in allowed.get(current, set()):
                raise InputError(f'Unsupported feedback transition: {current} -> {new_status}')
            entry['status'] = new_status
            entry['reviewed_at_utc'] = datetime.now(timezone.utc).isoformat()
            break
    if not found:
        raise InputError('Feedback record not found')
    with tempfile.NamedTemporaryFile('w', delete=False, dir=p.parent, encoding='utf-8') as tmp:
        for entry in entries:
            tmp.write(json.dumps(entry, ensure_ascii=False) + '\n')
        tmp_path = Path(tmp.name)
    os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, p)
    return {'id': record_id, 'status': new_status, 'reviewed': True}
