'use strict';
const $ = s => document.querySelector(s);
const state = { id:'', token:'', socket:null, matched:false, leaving:false, retry:0, pending:[], partnerPresent:false };
const hidden=(id,value)=>$(id).classList.toggle('hidden',value);
const put=(id,value)=>{$(id).textContent=String(value)};
const notify=(message,bad=false)=>{put('#notice',message);hidden('#notice',!message);$('#notice').classList.toggle('info',!bad);$('#notice').setAttribute('role',bad?'alert':'status')};
const updatePresence=(message,matched)=>{state.matched=matched;put('#presence-text',message);$('#message').disabled=!matched;$('#send').disabled=!matched;$('#message').placeholder=matched?'Say hello…':'Looking for someone new…'};
function emptyThread(heading,description) {
  $('#messages').replaceChildren();
  const card=document.createElement('div');card.className='empty-chat';card.id='empty-chat';
  const ico=document.createElement('div');ico.className='empty-illustration';ico.textContent='✧';
  const h=document.createElement('h2');h.textContent=heading;
  const p=document.createElement('p');p.textContent=description;
  card.append(ico,h,p);$('#messages').append(card);
  put('#typing','');
}
function bubble(mine,value){
  $('#empty-chat')?.remove();
  const el=document.createElement('div');el.className='bubble'+(mine?' mine':'');
  const who=document.createElement('div');who.className='bubble-head';who.textContent=mine?'You':'Stranger';
  const body=document.createElement('div');body.className='bubble-body';body.textContent=value;
  const time=document.createElement('span');time.className='bubble-time';time.textContent=new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
  el.append(who,body,time);$('#messages').append(el);$('#messages').scrollTop=$('#messages').scrollHeight;
}
function processPacket(msg) {
  let p;try{p=JSON.parse(msg.data)}catch{return}
  switch(p.type){
    case 'ready': updatePresence('Looking for your match…',false);break;
    case 'searching': updatePresence('Waiting for another stranger to join',false);break;
    case 'matched': emptyThread('Say hello to someone new','You are now connected to a random stranger.');updatePresence('A stranger is connected · Text only',true);notify('');$('#message').focus();break;
    case 'partner_left':emptyThread('Finding your next conversation…','Your previous stranger left. You are being matched again.');updatePresence('Looking for another stranger…',false);state.pending=[];break;
    case 'message':bubble(false,p.text);put('#typing','');break;
    case 'sent':if(state.pending.length)bubble(true,state.pending.shift());break;
    case 'typing':put('#typing',p.active?'Stranger is typing…':'');break;
    case 'reported':notify(p.message||'Report recorded for this temporary session.');break;
    case 'error':notify(p.message||'Something went wrong.',true);state.pending.shift();break;
  }
}
function connect() {
  if(state.leaving||!state.id)return;
  const ws=new WebSocket((location.protocol==='https:'?'wss:':'ws:')+'//'+location.host+'/ws');
  state.socket=ws;
  ws.addEventListener('open',()=>{state.retry=0;ws.send(JSON.stringify({type:'join',session_id:state.id,token:state.token}))});
  ws.addEventListener('message',processPacket);
  ws.addEventListener('close',e=>{
    if(state.socket!==ws||state.leaving)return;
    updatePresence('Connection lost',false);
    state.pending=[];
    // Temporary token can resume if the server has not restarted. Avoid unbounded background retries.
    if(state.retry<4){const delay=1000*2**state.retry++;setTimeout(connect,delay)}
    else notify('This session has ended or the free server restarted. Start a new conversation.',true);
  });
}
async function start(){
  if(!$('#adult-confirmed').checked||!$('#rules-accepted').checked){notify('Please confirm you are 18+ and accept the chat rules.',true);return}
  $('#find-stranger').disabled=true;notify('');
  try{
    const response=await fetch('/api/sessions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({adult_confirmed:true,rules_accepted:true})});
    const info=await response.json();if(!response.ok)throw Error(info.detail||'Could not start chat.');
    state.id=info.session_id;state.token=info.token;state.leaving=false;state.retry=0;
    hidden('#hero',true);hidden('#chat-screen',false);hidden('#reassurance',true);
    emptyThread('Finding someone new…','Another stranger will appear when they join the waiting queue.');
    updatePresence('Searching for a stranger…',false);
    connect();
  }catch(error){notify(error.message,true)}
  finally{$('#find-stranger').disabled=false}
}
function socketSend(data){
  if(!state.socket||state.socket.readyState!==WebSocket.OPEN){notify('Not connected. Try again.',true);return false}
  state.socket.send(JSON.stringify(data));return true;
}
function switchStranger(reportReason) {
  if(!state.matched){notify('Still waiting for a stranger to connect.',false);return}
  if(socketSend(reportReason?{type:'report',reason:reportReason}:{type:'next'})){
    state.pending=[];emptyThread('Looking for another stranger…','You will not be immediately paired with the stranger you skipped.');
    updatePresence('Finding someone else…',false);
  }
}
$('#find-stranger').addEventListener('click',start);
$('#next-stranger').addEventListener('click',()=>switchStranger());
$('#report-stranger').addEventListener('click',()=>{
  if(!state.matched){notify('You can report someone after a match.',true);return}
  const answer=prompt('Report reason: harassment, sexual, hate, threats, spam, other');
  if(answer===null)return;
  const reason=answer.trim().toLowerCase();
  if(!['harassment','sexual','hate','threats','spam','other'].includes(reason)){notify('Please enter one of the listed reasons.',true);return}
  if(confirm('Skip this stranger and submit a temporary report? The beta has no active report-review team.'))switchStranger(reason);
});
$('#leave-room').addEventListener('click',()=>{state.leaving=true;socketSend({type:'leave'});state.socket?.close();location.assign('/')});
$('#composer').addEventListener('submit',event=>{
  event.preventDefault();if(!state.matched)return;
  const content=$('#message').value.trim();if(!content||content.length>1000)return;
  if(socketSend({type:'message',text:content})){state.pending.push(content);$('#message').value=''}
});
$('#message').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();$('#composer').requestSubmit()}});
let lastTyping=0;$('#message').addEventListener('input',()=>{const now=Date.now();if(state.matched&&now-lastTyping>1200){lastTyping=now;socketSend({type:'typing',active:true})}});
async function loadAdviceConfig(){
  try{
    const response=await fetch('/api/advice/config');if(!response.ok)throw Error('Unavailable');
    const data=await response.json();
    if(data.open){
      put('#advice-price',new Intl.NumberFormat('en-NG',{style:'currency',currency:'NGN',maximumFractionDigits:0}).format(data.price_kobo/100)+' · '+data.minutes+' minutes');
      $('#pay-session').disabled=false;put('#booking-status','Bookings are subject to host confirmation. Payment is verified before a session is scheduled.');
    } else {
      put('#advice-price','Bookings not open yet');
      $('#pay-session').disabled=true;
    }
  }catch{$('#pay-session').disabled=true;put('#booking-status','Advice bookings are temporarily unavailable. The free stranger chat remains open.')}
}
$('#booking-form').addEventListener('submit',async event=>{
  event.preventDefault();const button=$('#pay-session');if(button.disabled)return;
  button.disabled=true;put('#booking-status','Opening secure checkout…');
  try{
    const response=await fetch('/api/advice/bookings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
      email:$('#booking-email').value.trim(),topic:$('#booking-topic').value.trim(),
      adult_confirmed:$('#booking-adult').checked
    })});
    const data=await response.json();
    if(!response.ok)throw Error(data.detail||'Checkout unavailable');
    if(!/^https:\/\/checkout\.paystack\.com\//.test(data.checkout_url))throw Error('Checkout URL could not be verified.');
    location.assign(data.checkout_url);
  }catch(error){put('#booking-status',error.message);button.disabled=false}
});
const params=new URLSearchParams(location.search);
if(params.get('reference')){put('#booking-status','Returned from checkout. Payment must be verified by the server; a return alone is not proof of payment.');history.replaceState(null,'','/')}
loadAdviceConfig();
