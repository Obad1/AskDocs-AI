"""End-to-end HTTP tests for eighteen distinct synthetic review workspaces."""
from __future__ import annotations
import csv
from io import BytesIO
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from product_app.api import create_app
from product_app.catalog import PRODUCTS
from product_app.store import Store

ROOT = Path(__file__).resolve().parents[1]
PASSWORD = 'Test_Password_Long_2026!'
SECRET = 'a-b-c-d-local-test-operator-cookie-signature-key-2026'


@pytest.fixture
def local(tmp_path):
    path = str(tmp_path / 'private' / 'operations.sqlite3')
    app = create_app(db_path=path, admin_password=PASSWORD, session_secret=SECRET, allow_leads=True)
    with TestClient(app) as client:
        yield client, path


def login(client):
    response = client.post('/api/auth/login', json={'password': PASSWORD})
    assert response.status_code == 200
    return {'X-CSRF-Token': response.json()['csrf']}


def uploads_for(code):
    if code == 'AA02':
        return {'bank': ('bank_synthetic.csv', (ROOT / 'samples/bank_synthetic.csv').read_bytes(), 'text/csv'),
                'ledger': ('ledger_synthetic.csv', (ROOT / 'samples/ledger_synthetic.csv').read_bytes(), 'text/csv')}
    file = 'freight_synthetic.csv' if code == 'AA03' else code + '.csv'
    return {'data': (file, (ROOT / 'samples' / file).read_bytes(), 'text/csv')}


def create_run(client, code, headers):
    response = client.post('/api/runs/' + code, data={'synthetic':'true','as_of':'2026-11-07','date_tolerance':'1'}, files=uploads_for(code), headers=headers)
    assert response.status_code == 200, (code, response.text)
    return response.json()


def test_public_catalog_has_all_and_only_18_real_engines(local):
    c, _ = local
    response = c.get('/api/products')
    assert response.status_code == 200
    products = response.json()
    assert len(products) == 18
    assert {p['id'] for p in products} == {f'AA{i:02}' for i in range(1,19)}
    assert all(p['columns'] and p['inputs'] and p['deliverable'] and p['limitations'] for p in products)
    assert all(p['production_certified'] is False and p['human_review_required'] for p in products)
    for p in products:
        assert c.get('/products/' + p['id']).status_code == 200
    assert c.get('/products/AA19').status_code == 404
    assert c.get('/assets/app.js').status_code == 200
    assert c.get('/assets/style.css').status_code == 200
    assert 'SYNTHETIC DATA ONLY' in c.get('/').text


def test_flagship_four_disclose_pain_proof_and_release_limits(local):
    c, _ = local
    products = {item['id']: item for item in c.get('/api/products').json()}
    assert {code for code, item in products.items() if item['featured']} == {'AA01','AA02','AA03','AA04'}
    for code in ('AA01','AA02','AA03','AA04'):
        item = products[code]
        assert all(item[field] for field in ('pain','approach','proof','buyer_question','priority_label'))
        assert item['human_review_required'] and item['demo_only']
        assert not item['production_certified']
        assert item['inputs'] and item['columns']
    for code in ('AA05','AA06','AA07','AA08','AA09','AA10','AA11','AA12','AA13','AA14','AA15','AA16','AA17','AA18'):
        assert not products[code]['featured']
    html = c.get('/').text
    assert 'buyer-fit' in html and 'buyer-pain' in html and 'buyer-proof' in html


def test_auth_failure_and_csrf_rules(local):
    c, _ = local
    assert c.get('/api/runs').status_code == 401
    assert c.post('/api/auth/login', json={'password':'wrong'}).status_code == 401
    headers = login(c)
    assert c.get('/api/runs').status_code == 200
    assert c.post('/api/runs/AA02', data={'synthetic':'true'}, files=uploads_for('AA02')).status_code == 403
    assert c.post('/api/runs/AA02', data={'synthetic':'true'}, files=uploads_for('AA02'),headers={'X-CSRF-Token':'made-up'}).status_code == 403
    assert c.get('/api/auth/me').json()['authenticated']
    assert c.post('/api/auth/logout', headers=headers).status_code == 200
    assert c.get('/api/runs').status_code == 401


def test_all_18_products_have_executable_http_workflow_and_saved_case(local):
    c, db = local
    headers = login(c)
    ids = set()
    for code in sorted(PRODUCTS):
        result = create_run(c, code, headers)
        assert result['findings'] > 0
        assert result['product'] == code
        assert result['synthetic'] is True
        assert result['production_certified'] is False
        ids.add(result['id'])
        detail = c.get('/api/runs/' + result['id']).json()
        assert detail['synthetic'] is True
        assert detail['state'] == 'in_review'
        assert len(detail['results']) == len(detail['decisions']) == result['findings']
        assert all(f['review_required'] for f in detail['results'])
        assert all(d['state'] == 'pending' for d in detail['decisions'])
        audit = c.get('/api/runs/' + result['id'] + '/history').json()
        assert audit[0]['action'] == 'created'
        assert len(c.get('/api/runs?product='+code).json()) == 1
    assert len(ids) == 18
    assert len(c.get('/api/runs').json()) == 18
    # Data must survive an application restart (single-instance SQLite).
    assert len(Store(db).list_runs()) == 18


def test_aa02_reviewer_lifecycle_history_and_exports(local):
    c, db = local
    auth = login(c)
    run = create_run(c, 'AA02', auth)
    identifier = run['id']
    detail = c.get('/api/runs/'+identifier).json()
    assert any(r['status'] == 'SUGGESTED_MATCH' for r in detail['results'])
    assert any('AMBIGUOUS' in r['status'] for r in detail['results'])
    assert len(detail['results']) >= 10
    for index in range(len(detail['results'])):
        status = 'accepted' if detail['results'][index]['status']=='SUGGESTED_MATCH' else 'rejected'
        update = c.post(f'/api/runs/{identifier}/findings/{index}',
                        json={'status':status,'note':'Checked against invented sample '+str(index),'reviewer':'test reviewer'}, headers=auth)
        assert update.status_code == 200, update.text
    complete = c.get('/api/runs/'+identifier).json()
    assert complete['state']=='review_complete'
    assert all(d['state'] in ('accepted','rejected') for d in complete['decisions'])
    assert len(c.get('/api/runs/'+identifier+'/history').json()) == len(complete['results'])+1
    csv_export=c.get('/api/runs/'+identifier+'/export.csv')
    assert csv_export.status_code==200
    assert 'review_status' in csv_export.text
    assert 'accepted' in csv_export.text
    xlsx=c.get('/api/runs/'+identifier+'/export.xlsx')
    assert xlsx.status_code==200
    workbook=load_workbook(BytesIO(xlsx.content), read_only=True)
    assert workbook['Read me']['B1'].value == 'SYNTHETIC-ONLY / DEMONSTRATION'
    assert workbook['Human review'].max_row == len(complete['results'])+1
    assert c.get('/api/runs/'+identifier+'/export.json').json()['state']=='review_complete'
    assert Store(db).get_run(identifier)['state']=='review_complete'


def test_only_dummy_data_allowed_and_invalid_input_is_not_saved(local):
    c, _ = local
    auth=login(c)
    denied=c.post('/api/runs/AA02', data={'synthetic':'false'},files=uploads_for('AA02'),headers=auth)
    assert denied.status_code==400
    bad_csv=b'id,date,amount,currency,reference\na,2026-11-07,10,NGN,X\n A ,2026-11-07,10,NGN,X\n'
    denied=c.post('/api/runs/AA02',data={'synthetic':'true'},files={**uploads_for('AA02'),'bank':('bad.csv',bad_csv,'text/csv')},headers=auth)
    assert denied.status_code==400
    assert 'duplicate id' in denied.json()['detail']
    denied=c.post('/api/runs/AA14',data={'synthetic':'true'}, files={'data':('phish.csv',b'id,wrong\nX,1\n','text/csv')},headers=auth)
    assert denied.status_code==400
    assert c.get('/api/runs').json()==[]


def test_all_sample_downloads_are_only_accessible_to_staff(local):
    c, _=local
    for code in PRODUCTS:
        for slot in PRODUCTS[code]['inputs']:
            assert c.get(f'/api/samples/{code}/{slot}').status_code==401
    login(c)
    for code in PRODUCTS:
        for slot in PRODUCTS[code]['inputs']:
            result=c.get(f'/api/samples/{code}/{slot}')
            assert result.status_code==200
            assert b'id,' in result.content


def test_no_raw_uploaded_files_retained(local):
    c, path=local
    auth=login(c)
    result=create_run(c, 'AA03', auth)
    stored=c.get('/api/runs/'+result['id']).json()
    assert set(stored['input_hashes'])=={'data'}
    assert not any(f.suffix.lower()=='.csv' for f in Path(path).parent.rglob('*') if f.is_file())
    assert 'data' not in stored['source_names'] or stored['source_names']['data'].endswith('.csv')


def test_enquiry_intake_with_consent_is_private_and_no_email_sent(local):
    c, _=local
    payload={'product':'AA02','organization':'Invented Synthetic Co','contact_email':'demo@example.test',
             'problem':'We need an entirely fictional review of invented ledger samples.', 'consent':True,'website':''}
    denied=c.post('/api/enquiries',json={**payload,'consent':False})
    assert denied.status_code==400
    accepted=c.post('/api/enquiries',json=payload)
    assert accepted.status_code==200
    assert 'no automatic email' in accepted.json()['note'].lower()
    assert c.get('/api/enquiries').status_code==401
    auth=login(c)
    enquiry=c.get('/api/enquiries').json()
    assert len(enquiry)==1 and enquiry[0]['status']=='new'
    assert c.post('/api/enquiries/'+accepted.json()['id']+'/status',json={'status':'qualified'},headers=auth).status_code==200
    assert c.get('/api/enquiries').json()[0]['status']=='qualified'


def test_enquiries_disabled_by_default(tmp_path):
    app=create_app(db_path=str(tmp_path/'private.sqlite3'),admin_password=PASSWORD,session_secret=SECRET,allow_leads=False)
    with TestClient(app) as c:
        r=c.post('/api/enquiries',json={'product':'AA02','organization':'Not Real','contact_email':'test@example.org','problem':'Fictional task, with enough characters.','consent':True})
        assert r.status_code==503


def test_csv_export_protects_formula_injection(local):
    c, _=local
    auth=login(c)
    bank=b'id,date,amount,currency,reference\nB1,2026-11-07,50,NGN,"=HYPERLINK(1)"\n'
    ledger=b'id,date,amount,currency,reference\nL1,2026-11-07,50,NGN,"=HYPERLINK(1)"\n'
    files={'bank':('bank.csv',bank,'text/csv'),'ledger':('ledger.csv',ledger,'text/csv')}
    r=c.post('/api/runs/AA02',data={'synthetic':'true','date_tolerance':'0'},files=files,headers=auth)
    assert r.status_code==200, r.text
    download=c.get('/api/runs/'+r.json()['id']+'/export.xlsx')
    assert download.status_code==200
    wb=load_workbook(BytesIO(download.content),read_only=True)
    assert all(cell.data_type!='f' for row in wb['Human review'].iter_rows() for cell in row)


def test_wrong_workflow_and_invalid_decision_are_rejected(local):
    c, _=local
    auth=login(c)
    assert c.post('/api/runs/AA99', data={'synthetic':'true'},files=uploads_for('AA01'),headers=auth).status_code==404
    result=create_run(c,'AA01',auth)
    identifier=result['id']
    assert c.post('/api/runs/'+identifier+'/findings/9999',json={'status':'accepted'},headers=auth).status_code==400
    assert c.post('/api/runs/'+identifier+'/findings/0',json={'status':'payed'},headers=auth).status_code==400
    assert c.get('/api/runs/this-does-not-exist').status_code==404


def test_security_headers_and_health(local):
    c,_=local
    r=c.get('/')
    assert r.headers['X-Content-Type-Options']=='nosniff'
    assert r.headers['X-Frame-Options']=='DENY'
    assert "script-src 'self'" in r.headers['Content-Security-Policy']
    assert c.get('/healthz').json()=={'ok':True,'mode':'synthetic-only','products':18}
