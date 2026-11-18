'use strict';
const el = id => document.getElementById(id);
let accessKey = '';
const feedback = value => { el('moderation-feedback').textContent = String(value); };
function line(parent,label,value) {
  const p=document.createElement('p'), b=document.createElement('strong');
  b.textContent=label+': ';p.append(b,document.createTextNode(String(value||'—')));parent.append(p);
}
async function refresh() {
  accessKey=el('moderator-secret').value;
  if(accessKey.length<32){feedback('Enter an authorized access key.');return;}
  el('refresh-reports').disabled=true;
  feedback('Loading report queue…');
  try {
    const res=await fetch('/api/moderation/reports?status='+encodeURIComponent(el('report-status-filter').value),
      {headers:{'X-Talk-Moderator-Secret':accessKey},cache:'no-store'});
    const data=await res.json();
    if(!res.ok)throw new Error(data.detail||'Report queue unavailable');
    const root=el('moderation-results');root.replaceChildren();
    for(const report of data.reports||[]) {
      const card=document.createElement('article');card.className='start-card';
      line(card,'Receipt',report.receipt);line(card,'Reason',report.reason);
      line(card,'Status',report.status);line(card,'Received',report.created_at);
      const field=document.createElement('select');
      field.setAttribute('aria-label','Review status for '+report.receipt);
      for(const status of ['reviewing','resolved','dismissed']) {
        const opt=document.createElement('option');opt.value=status;opt.textContent=status;
        opt.selected=report.status===status;field.append(opt);
      }
      const note=document.createElement('textarea');note.maxLength=500;note.rows=2;
      note.placeholder='Brief action summary (avoid personal details)';
      note.value=report.review_note||'';
      note.setAttribute('aria-label','Review note for '+report.receipt);
      const button=document.createElement('button');button.type='button';button.className='secondary-button';
      button.textContent='Save review';
      button.addEventListener('click',async()=>{
        button.disabled=true;
        try{
          const reply=await fetch('/api/moderation/reports/'+encodeURIComponent(report.receipt)+'/review',{
            method:'POST',
            headers:{'Content-Type':'application/json','X-Talk-Moderator-Secret':accessKey},
            body:JSON.stringify({status:field.value,note:note.value})
          });
          const body=await reply.json();
          if(!reply.ok)throw new Error(body.detail||'Review not saved');
          feedback('Review saved for '+report.receipt);
          await refresh();
        }catch(e){feedback(e.message)}finally{button.disabled=false}
      });
      card.append(field,note,button);root.append(card);
    }
    if(!(data.reports||[]).length)root.textContent='No reports in this filter.';
    feedback((data.reports||[]).length+' report records loaded. Manual queue; not continuously monitored.');
  }catch(err) {
    el('moderation-results').replaceChildren();
    feedback(err.message);
  }finally{el('refresh-reports').disabled=false}
}
el('refresh-reports').addEventListener('click',refresh);
el('clear-access').addEventListener('click',()=>{
  accessKey='';el('moderator-secret').value='';el('moderation-results').replaceChildren();
  feedback('Access cleared from this page.');
});
window.addEventListener('pagehide',()=>{accessKey='';el('moderator-secret').value='';});
