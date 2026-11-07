'use strict';
const $ = selector => document.querySelector(selector);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = String(text);
  return node;
};
const state = { catalog:[], selected:'AA02', csrf:null, currentRun:null };
function alertMessage(message, isError=false) {
  const box=$('#alerts');box.replaceChildren();
  if(message)box.append(el('div','notice'+(isError?' error':''),message));
}
async function call(url, options={}) {
  const response=await fetch(url,{credentials:'same-origin',...options});
  let body;
  try {body=await response.json();} catch {body={detail:'Unable to read the server response'};}
  if(!response.ok)throw new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail||body));
  return body;
}
function selectedProduct(){return state.catalog.find(p=>p.id===state.selected);}
function nodeLabel(text){const b=el('strong','',text);return b;}
function setSelected(code, push=true){
  if(!state.catalog.some(p=>p.id===code))return;
  state.selected=code;
  if(push)history.pushState({},'','/products/'+code);
  renderCatalog();renderProduct();loadRuns();
}
function renderCatalog(){
  const term=$('#search').value.trim().toLowerCase();
  const nav=$('#product-nav');nav.replaceChildren();
  for(const p of state.catalog){
    if(![p.id,p.name,p.sector,p.buyer].join(' ').toLowerCase().includes(term))continue;
    const button=el('button','nav-item'+(p.id===state.selected?' active':''));
    button.type='button';button.setAttribute('aria-current',p.id===state.selected?'page':'false');
    button.append(el('b','',p.id),el('span','',p.name));
    button.addEventListener('click',()=>setSelected(p.id));nav.append(button);
  }
  $('#count').textContent=state.catalog.length;
}
function renderProduct(){
  const p=selectedProduct();if(!p)return;
  $('#sector').textContent=p.sector;
  $('#code').textContent=p.id;
  $('#product-name').textContent=p.name;
  $('#product-job').textContent=p.job;
  $('#deliverable').textContent=p.deliverable;
  $('#limitations').textContent=p.limitations;
  $('#tol-field').classList.toggle('hidden',p.id!=='AA02');
  $('#date-field').classList.toggle('hidden',p.id==='AA02'||p.id==='AA03');
  const fields=$('#upload-fields'); fields.replaceChildren();
  p.inputs.forEach((slot,i)=>{
    const group=el('label','input-file');
    group.append(nodeLabel(slot==='data'?'Structured evidence file':slot==='bank'?'Bank statement export':'Ledger export'));
    const input=el('input');input.type='file';input.name=slot;input.id='file-'+slot;input.accept='.csv,.xlsx';
    input.setAttribute('aria-label',slot+' structured export');group.append(input);fields.append(group);
  });
  const schema=$('#schemas');schema.replaceChildren();
  p.columns.forEach((columns,i)=>{const card=el('div','schema');card.append(el('b','',p.inputs[i]+' · required CSV headings'),el('code','',columns));schema.append(card)});
  $('#synthetic').checked=false;
  $('#run-form').reset();
  $('#enquiry-product').value=p.id;
  $('#findings-section').classList.add('hidden');state.currentRun=null;
}
function renderRunList(runs){
  const target=$('#run-list');target.replaceChildren();
  if(!runs.length){target.append(el('p','muted','No saved synthetic reviews for this product. Run a sample to create a case.'));return;}
  for(const r of runs){
    const row=el('button','run-card');row.type='button';
    const left=el('span');left.append(el('strong','',r.title),el('small','',r.created_at.replace('T',' ').replace('+00:00',' UTC')+' · '+r.id.slice(0,8)));
    row.append(left,el('span','state',r.state.replace('_',' ')));
    row.addEventListener('click',()=>openRun(r.id));target.append(row);
  }
}
async function loadRuns(){
  if(!state.csrf){$('#run-list').textContent='Sign in to see your saved synthetic cases.';return}
  try{const runs=await call('/api/runs?product='+encodeURIComponent(state.selected));renderRunList(runs)}
  catch(err){$('#run-list').textContent=err.message;}
}
async function checkAuth(){
  try{const user=await call('/api/auth/me');state.csrf=user.csrf;$('#toggle-login').textContent='Sign out';$('#login-panel').classList.add('hidden');await loadRuns();}
  catch{state.csrf=null;$('#toggle-login').textContent='Operator sign in';$('#run-list').textContent='Sign in to see your saved synthetic cases.';}
}
async function login(event){
  event.preventDefault();
  try{const user=await call('/api/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:$('#password').value})});
    state.csrf=user.csrf;$('#password').value='';$('#login-panel').classList.add('hidden');$('#toggle-login').textContent='Sign out';alertMessage('Operator signed in. Only synthetic input may be processed.');await loadRuns();}
  catch(err){alertMessage(err.message,true)}
}
async function signInOut(){
  if(state.csrf){try{await call('/api/auth/logout',{method:'POST',headers:{'X-CSRF-Token':state.csrf}})}catch{}
    state.csrf=null;state.currentRun=null;$('#findings-section').classList.add('hidden');$('#toggle-login').textContent='Operator sign in';await loadRuns();alertMessage('Signed out.');}
  else $('#login-panel').classList.toggle('hidden');
}
async function runFiles(files){
  if(!state.csrf){alertMessage('Sign in before processing a sample.',true);$('#login-panel').classList.remove('hidden');return}
  if(!$('#synthetic').checked){alertMessage('Confirm that your uploaded files are wholly invented synthetic records.',true);return}
  const data=new FormData();data.set('synthetic','true');
  data.set('date_tolerance',$('#date-tolerance').value);data.set('as_of',$('#as-of').value);
  for(const [slot,file] of Object.entries(files))data.set(slot,file,file.name);
  const button=$('#run-button');button.disabled=true;
  try{
    const created=await call('/api/runs/'+state.selected,{method:'POST',headers:{'X-CSRF-Token':state.csrf},body:data});
    alertMessage(state.selected+' review completed. '+created.findings+' findings require human review.');
    await loadRuns();await openRun(created.id);
  }catch(err){alertMessage(err.message,true)}finally{button.disabled=false}
}
function getFormFiles(){
  const out={};for(const slot of selectedProduct().inputs){const input=$('#file-'+slot);if(!input.files.length)throw new Error('Choose the '+slot+' file');out[slot]=input.files[0]}return out;
}
async function submitRun(event){event.preventDefault();try{await runFiles(getFormFiles())}catch(err){alertMessage(err.message,true)}}
async function sampleRun(){
  if(!state.csrf){alertMessage('Sign in to load the included dummy data.',true);$('#login-panel').classList.remove('hidden');return;}
  $('#synthetic').checked=true;const files={};
  try{for(const slot of selectedProduct().inputs){const response=await fetch('/api/samples/'+state.selected+'/'+slot,{credentials:'same-origin'});
      if(!response.ok)throw new Error('Sample unavailable: '+slot);
      const blob=await response.blob();files[slot]=new File([blob],slot+'.csv',{type:'text/csv'});
    }await runFiles(files)}catch(err){alertMessage(err.message,true)}
}
function decisionForm(run,index){
  const current=run.decisions[index];const wrap=el('div','controls');
  const select=el('select');select.setAttribute('aria-label','Reviewer decision '+(index+1));
  for(const [value,label] of [['pending','Pending'],['accepted','Reviewed — accepted'],['rejected','Reviewed — rejected'],['needs_info','Needs clarification']]){
    const option=el('option','',label);option.value=value;option.selected=value===current.state;select.append(option);
  }
  const input=el('input');input.placeholder='Optional reviewer note';input.value=current.note;
  const save=el('button','btn btn-secondary','Save review');save.type='button';
  save.addEventListener('click',async()=>{try{await call('/api/runs/'+run.id+'/findings/'+index,{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':state.csrf},body:JSON.stringify({status:select.value,note:input.value,reviewer:'AVAMALE operator'})});alertMessage('Finding review saved with audit entry.');await openRun(run.id);await loadRuns()}catch(err){alertMessage(err.message,true)}});
  wrap.append(select,input,save);return wrap;
}
async function openRun(id){
  try{
    const run=await call('/api/runs/'+id);state.currentRun=run;
    $('#findings-section').classList.remove('hidden');$('#finding-count').textContent='· '+run.results.length;
    $('#run-subtitle').textContent='Synthetic only · '+run.id+' · '+run.state.replace('_',' ')+' · no automated decisions';
    const list=$('#findings');list.replaceChildren();
    run.results.forEach((result,index)=>{
      const card=el('article','finding');
      const title=el('h4');title.append(el('span','',String(index+1).padStart(2,'0')),el('span','',result.status||'REVIEW'));
      card.append(title);
      const preferred=['finding','rule','recommended_human_action','values','source_evidence','source_rows','bank_id','ledger_ids','record_id','shipment_id','source_evidence'];
      for(const key of [...new Set(preferred)])if(result[key]){card.append(el('p','',key.replaceAll('_',' ').toUpperCase()+': '+String(result[key])))}
      card.append(decisionForm(run,index));list.append(card);
    });
    const events=await call('/api/runs/'+id+'/history');const audit=$('#audit');audit.replaceChildren();
    events.forEach(e=>audit.append(el('div','',e.at+' · '+e.actor+' · '+e.action+' · '+e.detail)));
    $('#findings-section').scrollIntoView({behavior:'smooth',block:'start'});
  }catch(err){alertMessage(err.message,true)}
}
async function exportRun(ext){
  if(!state.currentRun)return;
  const response=await fetch('/api/runs/'+state.currentRun.id+'/export.'+ext,{credentials:'same-origin'});
  if(!response.ok){alertMessage('Could not export '+ext,true);return}
  const blob=await response.blob();const url=URL.createObjectURL(blob);const link=el('a');
  link.href=url;link.download=state.currentRun.product+'-review.'+ext;link.click();URL.revokeObjectURL(url);
}
async function sendEnquiry(event){
  event.preventDefault();const payload={product:$('#enquiry-product').value,organization:$('#organization').value,contact_email:$('#email').value,problem:$('#problem').value,consent:$('#contact-consent').checked,website:$('#hp').value};
  try{const value=await call('/api/enquiries',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});alertMessage('Enquiry recorded for manual review ('+value.id.slice(0,8)+'). No email has been sent.');$('#enquiry-form').reset()}
  catch(err){alertMessage(err.message,true)}
}
async function initialize(){
  try{state.catalog=await call('/api/products');
    if(state.catalog.length!==18)throw Error('Product catalog is incomplete');
    const path=location.pathname.split('/');if(path[1]==='products'&&state.catalog.some(p=>p.id===path[2]))state.selected=path[2];
    const picker=$('#enquiry-product');state.catalog.forEach(p=>{const option=el('option','',p.id+' · '+p.name);option.value=p.id;picker.append(option)});
    renderCatalog();renderProduct();await checkAuth();
  }catch(err){alertMessage('Could not load the product catalog: '+err.message,true)}
}
$('#search').addEventListener('input',renderCatalog);
$('#run-form').addEventListener('submit',submitRun);
$('#demo-button').addEventListener('click',sampleRun);
$('#login-form').addEventListener('submit',login);
$('#toggle-login').addEventListener('click',signInOut);
$('#refresh-runs').addEventListener('click',loadRuns);
$('#export-csv').addEventListener('click',()=>exportRun('csv'));
$('#export-json').addEventListener('click',()=>exportRun('json'));
$('#export-xlsx').addEventListener('click',()=>exportRun('xlsx'));
$('#enquiry-form').addEventListener('submit',sendEnquiry);
window.addEventListener('popstate',()=>{const path=location.pathname.split('/');const code=path[1]==='products'?path[2]:'AA02';setSelected(state.catalog.some(p=>p.id===code)?code:'AA02',false)});
initialize();
