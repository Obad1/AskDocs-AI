'use strict';
const $ = s => document.querySelector(s);
const state = { id:'', token:'', socket:null, matched:false, leaving:false, retry:0, pending:[], partnerPresent:false, everMatched:false, waitTimer:null, connectedOnce:false, reconnectTimer:null, exitTimer:null, chatStage:'home', reviewFocus:null };
const hidden=(id,value)=>$(id).classList.toggle('hidden',value);
const put=(id,value)=>{$(id).textContent=String(value)};
const notify=(message,bad=false)=>{put('#notice',message);hidden('#notice',!message);$('#notice').classList.toggle('info',!bad);$('#notice').setAttribute('role',bad?'alert':'status')};
const updatePresence=(message,matched)=>{state.matched=matched;put('#presence-text',message);$('#message').disabled=!matched;$('#send').disabled=!matched;$('#next-stranger').disabled=!matched;$('#report-stranger').disabled=!matched;$('#prompt-hello').disabled=!matched;$('#message').placeholder=matched?'Say hello…':'Looking for someone new…'; if(matched)state.chatStage='matched';};
function cancelWaitTimer(){ if(state.waitTimer){clearTimeout(state.waitTimer);state.waitTimer=null;} }
function showWaiting(){
  state.chatStage='waiting';
  cancelWaitTimer();
  updatePresence('Waiting for another person to join',false);
  hidden('#waiting-help',true);
  state.waitTimer=setTimeout(()=>{
    if(!state.matched&&!state.leaving){hidden('#waiting-help',false);put('#waiting-help-text','No one is available yet. You can leave and come back later; we won\'t invent a conversation or connect you to a bot.');}
  },18000);
}
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
    case 'searching': showWaiting();break;
    case 'matched': cancelWaitTimer();hidden('#waiting-help',true);state.everMatched=true;emptyThread('Say hello to someone new','You are now connected to a random stranger.');updatePresence('Connected · Text only',true);notify('');$('#message').focus();break;
    case 'resumed':cancelWaitTimer();hidden('#waiting-help',true);updatePresence('Reconnected to your conversation',true);notify('');break;
    case 'partner_reconnecting':state.chatStage='reconnecting';updatePresence('They lost connection · holding their seat for 12 seconds',false);break;
    case 'partner_left':emptyThread('Finding your next conversation…','The other person left. We are looking for somebody new.');showWaiting();state.pending=[];break;
    case 'message':bubble(false,p.text);put('#typing','');break;
    case 'sent':if(state.pending.length)bubble(true,state.pending.shift());break;
    case 'typing':put('#typing',p.active?'Stranger is typing…':'');break;
    case 'reported':notify((p.message||'Report submitted.')+(p.receipt?' Reference: '+p.receipt:''));break;
    case 'left':finishLeave();break;
    case 'error':notify(p.message||'Something went wrong.',true);state.pending.shift();break;
  }
}
function connect() {
  if(state.leaving||!state.id)return;
  const ws=new WebSocket((location.protocol==='https:'?'wss:':'ws:')+'//'+location.host+'/ws');
  state.socket=ws;
  ws.addEventListener('open',()=>ws.send(JSON.stringify({type:'join',session_id:state.id,token:state.token})));
  ws.addEventListener('message',event=>{
    let packet;try{packet=JSON.parse(event.data)}catch{return}
    if(packet.type==='error'&&/Session expired|Session expired or in use/.test(packet.message||'')){
      state.leaving=true;state.chatStage='reconnecting';state.pending=[];cancelWaitTimer();
      notify('Your temporary chat has expired. Use Start again to create a new session.',true);
      hidden('#restart-session',false);
      updatePresence('Session expired',false);
      return;
    }
    if(packet.type==='ready'){state.retry=0;state.connectedOnce=true;}
    processPacket(event);
  });
  ws.addEventListener('close',()=>{
    if(state.socket!==ws||state.leaving)return;
    state.chatStage='reconnecting';updatePresence('Connection interrupted · Trying to restore…',false);
    state.pending=[];
    cancelWaitTimer();
    if(state.retry<3){const delay=600*2**state.retry++;state.reconnectTimer=setTimeout(connect,delay)}
    else{state.leaving=true;notify('Connection could not be restored. Start a fresh conversation.',true);hidden('#restart-session',false)}
  });
  ws.addEventListener('error',()=>updatePresence('Network issue · Checking connection…',false));
}
async function start(){
  if(!$('#adult-confirmed').checked||!$('#rules-accepted').checked){notify('Please confirm you are 18+ and accept the chat rules.',true);return}
  $('#find-stranger').disabled=true;notify('');
  try{
    const response=await fetch('/api/sessions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({adult_confirmed:true,rules_accepted:true})});
    const info=await response.json();if(!response.ok)throw Error(info.detail||'Could not start chat.');
    state.id=info.session_id;state.token=info.token;state.leaving=false;state.retry=0;state.chatStage='waiting';
    hidden('#hero',true);hidden('#chat-screen',false);hidden('#reassurance',true);
    emptyThread('Finding someone new…','Another stranger will appear when they join the waiting queue.');
    showWaiting();
    hidden('#restart-session',true);
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
    showWaiting();
  }
}
$('#find-stranger').addEventListener('click',start);
$('#next-stranger').addEventListener('click',()=>switchStranger());
$('#report-stranger').addEventListener('click',()=>{
  if(!state.matched){notify('You can report someone after a match.',true);return}
  hidden('#report-form',false);
  $('#report-reason').focus();
});
$('#cancel-report').addEventListener('click',()=>hidden('#report-form',true));
$('#report-form').addEventListener('submit',event=>{
  event.preventDefault();
  const reason=$('#report-reason').value;
  if(!['harassment','sexual','hate','threats','spam','other'].includes(reason))return;
  hidden('#report-form',true);
  $('#report-reason').value='';
  switchStranger(reason);
});
$('#restart-session').addEventListener('click',()=>location.assign('/'));

$('#prompt-hello').addEventListener('click',()=>{
  if(!state.matched)return;
  const text='What is one small thing that made your day better?';
  $('#message').value=text;$('#message').focus();
});
function finishLeave(){
  clearTimeout(state.exitTimer);
  cancelWaitTimer();clearTimeout(state.reconnectTimer);
  state.socket?.close();
  location.assign('/');
}
function leaveChat(){
  if(state.leaving)return;
  state.leaving=true;state.matched=false;
  updatePresence('Leaving conversation…',false);
  $('#leave-room').disabled=true;
  const ws=state.socket;
  // Do not navigate before the server ACKs the intentional leave.
  // Without an ACK, fall back after a short timeout.
  if(ws?.readyState===WebSocket.OPEN){
    ws.send(JSON.stringify({type:'leave'}));
    state.exitTimer=setTimeout(finishLeave,1200);
  }else finishLeave();
}
$('#leave-room').addEventListener('click',leaveChat);
$('#end-wait').addEventListener('click',leaveChat);
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
    const published=Number.isInteger(data.price_kobo)&&data.price_kobo>0;
    if(published){
      const amount=new Intl.NumberFormat('en-NG',{style:'currency',currency:'NGN',maximumFractionDigits:0}).format(data.price_kobo/100);
      put('#advice-price',amount+' · '+data.minutes+' minutes · Live text');
    } else {
      put('#advice-price','Price to be announced');
    }
    $('#pay-session').disabled=!data.open;
    hidden('#booking-form',!data.open);
    hidden('#advice-interest',data.open);
    put('#booking-status',data.open
      ? 'Private advice only. Payment must be verified and your booking accepted before chatting.'
      : 'Bookings are not open yet. No payment will be taken.');
  }catch{$('#pay-session').disabled=true;hidden('#booking-form',true);hidden('#advice-interest',false);put('#booking-status','Advice bookings are temporarily unavailable. The free stranger chat remains open.')}
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
    if(data.reference&&data.customer_access_token)sessionStorage.setItem('talk-friend-paid-'+data.reference,data.customer_access_token);
    location.assign(data.checkout_url);
  }catch(error){put('#booking-status',error.message);button.disabled=false}
});
const bookingParams=new URLSearchParams(location.search);
const returnedReference=bookingParams.get('reference');
let currentPaid={reference:null,token:null,role:null,socket:null};
let hostSecret='';
const bookingKey=ref=>'talk-friend-paid-'+ref;
function renderPaidMessage(label,value){
  const card=document.createElement('div');card.className='bubble'+(label==='You'?' mine':'');
  const who=document.createElement('div');who.className='bubble-head';who.textContent=label;
  const content=document.createElement('div');content.className='bubble-body';content.textContent=String(value);
  card.append(who,content);$('#advice-messages').append(card);
  $('#advice-messages').scrollTop=$('#advice-messages').scrollHeight;
}
async function verifyMyBooking(){
  if(!currentPaid.reference||!currentPaid.token)return;
  put('#customer-booking-status','Verifying your payment securely…');
  try{
    const response=await fetch('/api/advice/bookings/'+encodeURIComponent(currentPaid.reference)+'/verify',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({access_token:currentPaid.token})
    });
    const body=await response.json();
    if(!response.ok)throw Error(body.detail||'Unable to verify booking');
    const status=body.paid?(body.provider_confirmed?'Payment confirmed. Your host accepted; enter your session.':'Payment confirmed. Waiting for the host to agree a session time.'):'Payment not confirmed. If you just paid, please try checking again.';
    put('#customer-booking-status',status);
    hidden('#enter-advice-chat',!body.provider_confirmed);
  }catch(error){put('#customer-booking-status',error.message)}
}
function enterPaidChat(reference,role,token){
  if(!reference||!token)return;
  currentPaid={reference,role,token,socket:null};
  hidden('#hero',true);hidden('#chat-screen',true);hidden('#advice-chat',false);hidden('#reassurance',true);
  hidden('#host-panel',true);
  $('#advice-messages').replaceChildren();
  put('#advice-presence','Connecting to your private session…');
  const scheme=location.protocol==='https:'?'wss:':'ws:';
  const ws=new WebSocket(scheme+'//'+location.host+'/ws/advice/'+encodeURIComponent(reference));
  currentPaid.socket=ws;
  ws.addEventListener('open',()=>ws.send(JSON.stringify({role,token})));
  ws.addEventListener('message',event=>{
    let p;try{p=JSON.parse(event.data)}catch{return}
    if(p.type==='joined')put('#advice-presence','Connected · Waiting for the other person');
    if(p.type==='connected')put('#advice-presence','Both people are here · Session live');
    if(p.type==='message')renderPaidMessage(role==='host'?'Customer':'Host',p.text);
    if(p.type==='sent')renderPaidMessage('You',p.text);
    if(p.type==='disconnected')put('#advice-presence','The other person disconnected');
  });
  ws.addEventListener('close',()=>put('#advice-presence','Session disconnected. You may re-enter while your booking is accepted.'));
}
$('#advice-composer').addEventListener('submit',event=>{
  event.preventDefault();const content=$('#advice-text').value.trim();
  if(!content||content.length>1000)return;
  const ws=currentPaid.socket;
  if(ws?.readyState!==WebSocket.OPEN){notify('Private chat is not connected.',true);return}
  ws.send(JSON.stringify({type:'message',text:content}));$('#advice-text').value='';
});
$('#close-advice').addEventListener('click',()=>{
  if(currentPaid.socket?.readyState===WebSocket.OPEN)currentPaid.socket.send(JSON.stringify({type:'end'}));
  currentPaid.socket?.close();location.assign('/');
});
$('#check-booking').addEventListener('click',verifyMyBooking);
$('#enter-advice-chat').addEventListener('click',()=>enterPaidChat(currentPaid.reference,'customer',currentPaid.token));
async function fetchHostBookings(){
  const secret=$('#host-secret').value;
  if(secret.length<32){put('#host-bookings','Enter the authorized host access key.');return}
  hostSecret=secret;put('#host-bookings','Loading…');
  try{
    const response=await fetch('/api/advice/host/bookings',{headers:{'X-Advice-Host-Secret':secret}});
    const body=await response.json();
    if(!response.ok)throw Error(body.detail||'Access denied');
    const target=$('#host-bookings');target.replaceChildren();
    for(const booking of body.bookings||[]){
      const card=document.createElement('article');card.className='host-booking';
      const heading=document.createElement('strong');heading.textContent=booking.reference+' · '+booking.status;
      const info=document.createElement('p');info.textContent=booking.customer_email+' · '+(booking.topic||'General advice');
      const accept=document.createElement('button');accept.type='button';accept.className='secondary-button';
      accept.textContent='Verify payment and accept';accept.disabled=!['pending','paid','accepted'].includes(booking.status);
      accept.addEventListener('click',async()=>{
        try{
          const res=await fetch('/api/advice/host/bookings/'+encodeURIComponent(booking.reference)+'/accept',{method:'POST',headers:{'X-Advice-Host-Secret':secret}});
          const v=await res.json();if(!res.ok)throw Error(v.detail||'Not confirmed');
          await fetchHostBookings();
        }catch(e){notify(e.message,true)}
      });
      const join=document.createElement('button');join.type='button';join.className='primary-button';
      join.textContent='Join paid advice';join.disabled=booking.status!=='accepted';
      join.addEventListener('click',()=>enterPaidChat(booking.reference,'host',secret));
      card.append(heading,info,accept,join);target.append(card);
    }
    if(!(body.bookings||[]).length)target.textContent='No paid bookings yet.';
  }catch(error){put('#host-bookings',error.message)}
}
$('#host-refresh').addEventListener('click',fetchHostBookings);
if(location.pathname==='/host'){
  hidden('#hero',true);hidden('#reassurance',true);hidden('#host-panel',false);
}
if(returnedReference&&/^ADVICE_[A-Za-z0-9]{24}$/.test(returnedReference)){
  hidden('#customer-booking',false);
  const token=sessionStorage.getItem(bookingKey(returnedReference));
  if(token){
    currentPaid.reference=returnedReference;currentPaid.token=token;
    verifyMyBooking();
  }else put('#customer-booking-status','Booking return received, but this browser has no access token. Contact the host with your reference for help.');
  history.replaceState(null,'','/');
}
loadAdviceConfig();

let reviewReturnFocus=null;
function openReview(){
  reviewReturnFocus=document.activeElement;
  hidden('#review-panel',false);
  document.body.classList.add('review-open');
  $('#review-rating').focus();
}
function closeReview(){
  hidden('#review-panel',true);
  document.body.classList.remove('review-open');
  reviewReturnFocus?.focus?.();
}
$('#open-review').addEventListener('click',openReview);
$('#open-review-top').addEventListener('click',openReview);
$('#close-review').addEventListener('click',closeReview);
$('#review-scrim').addEventListener('click',closeReview);
document.addEventListener('keydown',event=>{
  if($('#review-panel').classList.contains('hidden'))return;
  if(event.key==='Escape'){event.preventDefault();closeReview()}
  if(event.key==='Tab'){
    const items=[...$('#review-panel').querySelectorAll('button:not([disabled]),select:not([disabled])')];
    const first=items[0],last=items[items.length-1];
    if(event.shiftKey&&document.activeElement===first){event.preventDefault();last.focus()}
    else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first.focus()}
  }
});
$('#review-form').addEventListener('submit',async event=>{
  event.preventDefault();const button=$('#submit-review');
  button.disabled=true;put('#review-status','Sending your review…');
  try{
    const response=await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({rating:Number($('#review-rating').value),issue:$('#review-issue').value,
        stage:state.chatStage==='home'||state.chatStage==='waiting'||state.chatStage==='matched'||state.chatStage==='reconnecting'
          ? state.chatStage:'other'})});
    const result=await response.json();
    if(!response.ok)throw new Error(result.detail||'Review could not be sent.');
    put('#review-status','Thank you. Review received · '+result.receipt+'. This is a beta issue log, not a staffed support ticket.');
    $('#review-form').reset();
  }catch(error){put('#review-status',error.message)}
  finally{button.disabled=false}
});

async function refreshAvailability(){
  try{
    const response=await fetch('/api/match/status',{cache:'no-store'});
    if(!response.ok)throw Error('Not available');
    const data=await response.json();
    if(!$('#hero').classList.contains('hidden')){
      put('#home-availability',data.online>0
        ? 'People online: '+data.online+' · Your match depends on who is available'
        : 'Nobody online right now · You can be the first to join');
    }
    if(state.chatStage==='waiting'&&state.id&&!state.matched){
      // Count includes this visitor, so never imply the visitor matches themselves.
      const others=Math.max(0,Number(data.waiting||0)-1);
      put('#presence-text',others>0
        ? 'Other people are waiting · Checking for an eligible match'
        : 'Waiting for a real person to join');
    }
  }catch{
    if(!$('#hero').classList.contains('hidden'))put('#home-availability','Availability is temporarily unknown');
  }
}
refreshAvailability();
setInterval(refreshAvailability,10000);
