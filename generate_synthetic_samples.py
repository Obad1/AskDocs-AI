"""Deterministic synthetic fixtures. Contains no client records."""
from pathlib import Path
import csv
from avamale_suite.workflows import SPEC

DATA = {
'AA01': [
['E1','EVIDENCE','ACCESS_LOG','Approved sample policy excerpt','yes','policy_v2_page_4'],
['Q1','QUESTION','ACCESS_LOG','How are access events reviewed?','',''],
['Q2','QUESTION','ENCRYPTION_AT_REST','Is data encrypted at rest?','','']],
'AA04': [
['P1','PO','PO_100','SKU_A','VENDOR_A','5','100.00','NGN'],
['I1','INVOICE','PO_100','SKU_A','VENDOR_A','5','100.00','NGN'],
['R1','RECEIPT','PO_100','SKU_A','VENDOR_A','5','100.00','NGN'],
['P2','PO','PO_101','SKU_B','VENDOR_B','7','200','NGN'],
['I2','INVOICE','PO_101','SKU_B','VENDOR_B','6','200','NGN'],
['R2','RECEIPT','PO_101','SKU_B','VENDOR_B','7','200','NGN']],
'AA05': [
['T1','TICKET_001','Cannot reset password','LOW','ACCOUNT'],
['T2','TICKET_002','Customer portal outage','MEDIUM','SERVICE']],
'AA06': [
['D1','DISPUTE','DSP_1','TX_1','4000.00','NGN',''],
['E1','EVIDENCE','DSP_1','TX_1','4000.00','NGN','receipt_ref_01'],
['D2','DISPUTE','DSP_2','TX_2','4500.00','NGN','']],
'AA07': [
['L1','LEASE_01','2026-01-01','2027-12-31','2027-09-30','Notice requires written request','lease_01_page_8'],
['L2','LEASE_02','2026-01-01','2027-12-31','2028-04-01','','']],
'AA08': [
['C1','CONTRACTOR','NONE','LAGOS','PLUMBING','NORMAL','YES','YES'],
['R1','REQUEST','REQ_1','LAGOS','PLUMBING','NORMAL','',''],
['R2','REQUEST','REQ_2','LAGOS','ELECTRICAL','EMERGENCY','','']],
'AA09': [
['INV1','invoice_01','CUST_SYN_1','1250.00','NGN','7','YES','NO'],
['INV2','invoice_02','CUST_SYN_2','1500.00','NGN','14','YES','YES']],
'AA10': [
['BASE1','BASELINE','CLOUD_STORAGE','BASE_PERIOD','200.00','USD'],
['ACT1','ACTUAL','CLOUD_STORAGE','2026-11','400.00','USD'],
['ACT2','ACTUAL','UNLISTED_SERVICE','2026-11','500.00','USD']],
'AA11': [
['REQ1','REQUIREMENT','SUP_A','SUPPLIER_POLICY','','',''],
['E1','EVIDENCE','SUP_A','SUPPLIER_POLICY','policy_2027','YES','2027-12-31'],
['REQ2','REQUIREMENT','SUP_B','GHG_DATA','','','']],
'AA12': [
['A1','ALERT','ALERT_X1','UNUSUAL_ACTIVITY','',''],
['E1','EVIDENCE','ALERT_X1','BACKUP_LOG','evidence_token_1','YES'],
['A2','ALERT','ALERT_X2','MISSING_DOCUMENT','','']],
'AA13': [
['C1','DSAR_SYN_001','ACCESS','YES','2026-11-01','2026-12-01','authority_attestation_1'],
['C2','DSAR_SYN_002','ERASURE','NO','2026-11-01','2026-12-01','']],
'AA14': [
['M1','synthetic_message_1','DMARC_FAIL','1','0','mail_headers_ref_1'],
['M2','synthetic_message_2','PASS','0','0','mail_headers_ref_2']],
'AA15': [
['O1','SYN_WORKER_1','SYN_APP_1','ACCESS_REVOKE','YES','IT_APPROVAL_1'],
['O2','SYN_WORKER_1','SYN_LAPTOP_1','EQUIPMENT_RETURN','NO','']],
'AA16': [
['P1','PROVIDER_SYN_1','LICENCE','2027-01-31','issuer_reference_1','YES'],
['P2','PROVIDER_SYN_2','LICENCE','2026-01-01','issuer_reference_2','YES']],
'AA17': [
['I1','CASE_SYN_1','DOB','YES','form_field_dob'],
['I2','CASE_SYN_1','CONTACT','YES','form_field_contact'],
['I3','CASE_SYN_2','DOB','NO',''],
['I4','CASE_SYN_2','CONTACT','YES','form_field_contact']],
'AA18': [
['C1','CASE','AUTH_SYN_01','','',''],
['E1','EVIDENCE','AUTH_SYN_01','CLINICIAN_NOTE','synthetic_note_1','YES'],
['E2','EVIDENCE','AUTH_SYN_01','SERVICE_REQUEST','synthetic_request_1','YES'],
['E3','EVIDENCE','AUTH_SYN_01','COVERAGE_INFO','synthetic_coverage_1','YES'],
['C2','CASE','AUTH_SYN_02','','',''],
['E4','EVIDENCE','AUTH_SYN_02','SERVICE_REQUEST','synthetic_request_2','YES']],
}

def main():
    root=Path(__file__).parent/'samples';root.mkdir(parents=True,exist_ok=True)
    for code,rows in DATA.items():
        columns=SPEC[code][1].split(',')
        assert all(len(row)==len(columns) for row in rows),(code,len(rows[0]),len(columns))
        with (root/f'{code}.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(columns);w.writerows(rows)
    print(f'Wrote {len(DATA)} synthetic workflow fixtures')

if __name__=='__main__':main()
