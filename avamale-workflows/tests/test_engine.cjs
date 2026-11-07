const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const e=require('../review-engine.js');let base=path.join(__dirname,'..','samples');let read=n=>fs.readFileSync(path.join(base,n),'utf8');
let bank=e.transactions(read('bank_synthetic.csv')),ledger=e.transactions(read('ledger_synthetic.csv'));
let r=e.reconcile(bank,ledger,1);assert.deepEqual(r.filter(x=>x.status==='SUGGESTED_MATCH').map(x=>x.bank_id),['B001','B002']);assert.equal(r.length,16);assert.equal(r.filter(x=>x.status==='DUPLICATE_OR_AMBIGUOUS').length,3);
assert.throws(()=>e.transactions('id,date,amount,currency,reference\nA,2026-02-30,1,NGN,X'),/Invalid calendar/);
assert.throws(()=>e.transactions('id,date,amount,currency,reference\nA,2026-01-01,1.001,NGN,X'),/decimal/);
assert.throws(()=>e.reconcile(bank,ledger,8),/tolerance/);
assert.equal(e.cents('0.10')+e.cents('0.20'),e.cents('0.30'));
assert.deepEqual(e.parseCSV('a,b\r\n"one,two","say ""hi"""\r\n'),[['a','b'],['one,two','say "hi"']]);
assert.throws(()=>e.parseCSV('a,b\n"unclosed,x'),/Unclosed/);
assert.match(e.csv([{value:' =1+1'}]),/\' =1\+1/);
let f=e.freight(e.freightRows(read('freight_synthetic.csv')));assert.equal(f.length,9);assert.equal(f.filter(x=>x.status==='FIELD_DISCREPANCY').length,6);assert.equal(f.filter(x=>x.status==='MISSING_DOCUMENT').length,1);
assert.equal(e.freight(e.freightRows(read('freight_synthetic.csv')),'10').filter(x=>x.field==='gross_weight_kg').length,0);
// Compare complete status/identity/field outcomes against the Python engine's generated fixtures.
let py=JSON.parse(fs.readFileSync(path.join(__dirname,'..','demo_outputs','AA02','review.json')));assert.deepEqual(r.map(x=>[x.status,x.bank_id,x.ledger_ids]),py.map(x=>[x.status,x.bank_id,x.ledger_ids]));
let pf=JSON.parse(fs.readFileSync(path.join(__dirname,'..','demo_outputs','AA03','review.json')));assert.deepEqual(f.map(x=>[x.status,x.shipment_id,x.item_code,x.field]),pf.map(x=>[x.status,x.shipment_id,x.item_code,x.field]));
console.log('Browser engine assertions and Python/JS fixture parity passed.');
