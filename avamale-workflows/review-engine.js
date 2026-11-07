'use strict';
(function(root){
const norm=v=>String(v??'').trim().replace(/\s+/g,' ').toUpperCase();
function parseCSV(text){
  text=String(text).replace(/^\uFEFF/,'');let rows=[],row=[],s='',quoted=false;
  for(let i=0;i<text.length;i++){let c=text[i];if(quoted){if(c==='"'&&text[i+1]==='"'){s+='"';i++;}else if(c==='"')quoted=false;else s+=c;}
    else if(c==='"'){if(s)throw Error('Unexpected quote in CSV');quoted=true;}else if(c===','){row.push(s);s='';}
    else if(c==='\n'||c==='\r'){if(c==='\r'&&text[i+1]==='\n')i++;row.push(s);rows.push(row);row=[];s='';}else s+=c;}
  if(quoted)throw Error('Unclosed quote in CSV');if(s||row.length){row.push(s);rows.push(row);}return rows;
}
function table(text,required){
  let rows=parseCSV(text);if(!rows.length)throw Error('Empty CSV');let heads=rows.shift().map(x=>x.trim().toLowerCase());
  if(heads.some(x=>!x)||new Set(heads).size!==heads.length)throw Error('Blank or duplicate headers');
  for(const h of required)if(!heads.includes(h))throw Error('Missing column: '+h);
  if(rows.length>1000)throw Error('This review demo supports at most 1,000 rows per CSV');
  let ids=new Set(),out=[];
  rows.forEach((r,i)=>{if(r.every(x=>!x.trim()))return;if(r.length!==heads.length)throw Error('Column count differs on row '+(i+2));let v=Object.fromEntries(heads.map((h,j)=>[h,r[j]]));v._source_row=i+2;
    if(!v.id.trim()||ids.has(v.id))throw Error('Blank or duplicate ID on row '+(i+2));ids.add(v.id);out.push(v);});
  if(!out.length)throw Error('No data rows');return out;
}
function cents(value){let s=String(value).trim();if(!/^[+-]?\d+(?:\.\d{1,2})?$/.test(s))throw Error('Use a decimal amount with at most 2 places, no commas or currency symbols');let negative=s[0]==='-';s=s.replace(/^[+-]/,'');let [whole,part='']=s.split('.');return (negative?-1n:1n)*(BigInt(whole)*100n+BigInt(part.padEnd(2,'0')));}
function isoDay(s){if(!/^\d{4}-\d{2}-\d{2}$/.test(s))throw Error('Dates must be YYYY-MM-DD');let n=Date.parse(s+'T00:00:00Z');if(!Number.isFinite(n)||new Date(n).toISOString().slice(0,10)!==s)throw Error('Invalid calendar date: '+s);return n/86400000;}
function transactions(text){let rows=table(text,['id','date','amount','currency','reference']);for(const r of rows){r._amount=cents(r.amount);r._currency=norm(r.currency);r._reference=norm(r.reference);r._date=isoDay(r.date);if(!/^[A-Z]{3}$/.test(r._currency))throw Error('Currency must be a three-letter code');}return rows;}
function reconcile(bank,ledger,tol=0){
 if(!Number.isInteger(tol)||tol<0||tol>7)throw Error('Date tolerance must be 0–7 whole days');
 let candidates=bank.map(b=>b._reference?ledger.flatMap((l,j)=>b._amount===l._amount&&b._currency===l._currency&&b._reference===l._reference&&Math.abs(b._date-l._date)<=tol?[j]:[]):[]);
 let reverse=ledger.map((_,j)=>candidates.flatMap((js,i)=>js.includes(j)?[i]:[]));let key=r=>[r._date,String(r._amount),r._currency,r._reference].join('\u0001');
 let dup=(arr,r)=>arr.filter(x=>key(x)===key(r)).length>1,used=new Set(),out=[];
 bank.forEach((b,i)=>{let js=candidates[i],unique=js.length===1&&reverse[js[0]].length===1&&!dup(bank,b)&&!dup(ledger,ledger[js[0]]);
  if(unique)used.add(js[0]);out.push({status:unique?'SUGGESTED_MATCH':(dup(bank,b)||js.length?'DUPLICATE_OR_AMBIGUOUS':(!b._reference?'MISSING_REFERENCE':'UNMATCHED_BANK')),bank_id:b.id,ledger_ids:js.map(j=>ledger[j].id).join(' | '),bank_source_row:b._source_row,ledger_source_rows:js.map(j=>ledger[j]._source_row).join(' | '),amount:b.amount,currency:b._currency,reference:b.reference,rule:unique?'Unique amount / currency / reference within explicit date window':'Review required; no automatic resolution',review_required:true});});
 ledger.forEach((l,j)=>{if(!used.has(j))out.push({status:reverse[j].length||dup(ledger,l)?'DUPLICATE_OR_AMBIGUOUS_LEDGER':(!l._reference?'MISSING_REFERENCE_LEDGER':'UNMATCHED_LEDGER'),bank_id:'',ledger_ids:l.id,bank_source_row:'',ledger_source_rows:l._source_row,amount:l.amount,currency:l._currency,reference:l.reference,rule:'Accountant review required; no posting',review_required:true});});return out;
}
function freightRows(text){let rows=table(text,['id','shipment_id','document_type','document_ref','item_code','quantity','packages','gross_weight_kg','consignee','origin_country']);for(const r of rows){r._type=norm(r.document_type);if(!['INVOICE','PACKING_LIST','BILL_OF_LADING'].includes(r._type))throw Error('Unsupported document_type');for(const k of ['shipment_id','document_ref','item_code','consignee','origin_country'])if(!r[k].trim())throw Error(k+' is required');for(const k of ['quantity','packages','gross_weight_kg']){r['_'+k]=cents(r[k]);if(r['_'+k]<=0n)throw Error(k+' must be positive');}if(r._packages%100n!==0n)throw Error('Packages must be whole numbers');if(!/^[A-Z]{2}$/.test(norm(r.origin_country)))throw Error('Origin country must be a two-letter code');}return rows;}
function freight(rows,tolerance='0'){
 const tol=cents(tolerance);if(tol<0n)throw Error('Weight tolerance cannot be negative');let groups=new Map();for(const r of rows){let k=norm(r.shipment_id)+'\u0001'+norm(r.item_code);if(!groups.has(k))groups.set(k,[]);groups.get(k).push(r);}let out=[];
 for(const [key,g] of groups){let [shipment_id,item_code]=key.split('\u0001'),counts={};g.forEach(r=>counts[r._type]=(counts[r._type]||0)+1);let base={shipment_id,item_code,source_evidence:g.map(r=>r._type+':'+r.document_ref+'@row'+r._source_row).join(' | '),review_required:true};let missing=['INVOICE','PACKING_LIST','BILL_OF_LADING'].filter(k=>!counts[k]),dups=Object.keys(counts).filter(k=>counts[k]>1),diff=0;
  if(missing.length)out.push({...base,status:'MISSING_DOCUMENT',field:'document_type',values:missing.sort().join(' | '),rule:'Required document missing for item'});
  if(dups.length)out.push({...base,status:'DUPLICATE_DOCUMENT_ITEM',field:'document_type',values:dups.sort().join(' | '),rule:'Amended / duplicate record requires review'});
  for(const f of ['quantity','packages','gross_weight_kg','consignee','origin_country']){let same;if(['quantity','packages','gross_weight_kg'].includes(f)){let v=g.map(r=>r['_'+f]),min=v.reduce((a,b)=>a<b?a:b),max=v.reduce((a,b)=>a>b?a:b);same=max-min<=(f==='gross_weight_kg'?tol:0n);}else same=new Set(g.map(r=>norm(r[f]))).size===1;
   if(!same){diff++;out.push({...base,status:'FIELD_DISCREPANCY',field:f,values:g.map(r=>r._type+'='+r[f]).join(' | '),rule:'Compare equivalent item-level fields / units'});}}
  if(!missing.length&&!dups.length&&!diff)out.push({...base,status:'CONSISTENT_FIELDS_REVIEW_REQUIRED',field:'',values:'',rule:'Fields agree; no authenticity or customs decision'});
 }return out;
}
function csv(rows){if(!rows.length)return '';let keys=Object.keys(rows[0]);let safe=v=>{let s=String(v??'');if(/^[\s]*[=+\-@]|^[\t\r\n]/.test(s))s="'"+s;return '"'+s.replace(/"/g,'""')+'"';};return [keys.map(safe).join(','),...rows.map(r=>keys.map(k=>safe(r[k])).join(','))].join('\r\n');}
const api={parseCSV,table,cents,transactions,reconcile,freightRows,freight,csv};if(typeof module!=='undefined')module.exports=api;else root.AVAMALE=api;
})(typeof window!=='undefined'?window:globalThis);
