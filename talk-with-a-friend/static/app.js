'use strict';

const $ = (selector) => document.querySelector(selector);
const state = {
  roomId: null, token: null, inviteUrl: '', seat: null, socket: null,
  friendOnline: false, intentionalClose: false, reconnectAttempts: 0,
  pending: [], name: 'Friend', typingTimer: null,
};
const CHAT_KEY = (id) => 'talkfriend-room-' + id;
const text = (element, value) => { element.textContent = String(value ?? ''); };
const hide = (element, hidden = true) => element.classList.toggle('hidden', hidden);
const notice = (value, isError = true) => {
  const box = $('#notice');
  text(box, value);
  box.classList.toggle('info', !isError);
  box.setAttribute('role', isError ? 'alert' : 'status');
  hide(box, !value);
};

async function requestJSON(url, init = {}) {
  const result = await fetch(url, { ...init, headers: { Accept: 'application/json', ...init.headers } });
  let data;
  try { data = await result.json(); } catch { data = {}; }
  if (!result.ok) throw new Error(data.detail || 'Connection failed. Please try again.');
  return data;
}
function roomPath(id) { return '/r/' + encodeURIComponent(id); }
function stored(id) {
  try { return JSON.parse(sessionStorage.getItem(CHAT_KEY(id)) || 'null'); } catch { return null; }
}
function store(id, value) { sessionStorage.setItem(CHAT_KEY(id), JSON.stringify(value)); }
function clearRoom() {
  if (state.roomId) sessionStorage.removeItem(CHAT_KEY(state.roomId));
  state.intentionalClose = true;
  state.socket?.close();
}
function updateConnectionStatus(message, online) {
  text($('#presence-text'), message);
  state.friendOnline = online;
  $('#message').disabled = !online;
  $('#send').disabled = !online;
  $('#message').placeholder = online ? "Write what's on your mind…" : 'Waiting for your friend to join…';
  if (online && document.activeElement !== $('#message')) $('#message').focus();
}
function appendBubble(kind, message, name = '') {
  hide($('#empty-chat'));
  const card = document.createElement('div');
  card.className = 'bubble' + (kind === 'mine' ? ' mine' : '');
  const who = document.createElement('span');
  who.className = 'bubble-head';
  text(who, kind === 'mine' ? 'You' : name || 'Your friend');
  const body = document.createElement('div');
  body.className = 'bubble-body';
  text(body, message); // DOM text only, never trust or parse user HTML.
  const time = document.createElement('span');
  time.className = 'bubble-time';
  text(time, new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}));
  card.append(who, body, time);
  $('#messages').append(card);
  $('#messages').scrollTop = $('#messages').scrollHeight;
}

function showChat() {
  hide($('#hero'));
  hide($('#join-panel'));
  hide($('#chat-screen'), false);
  hide($('#reassurance'));
  $('#leave-room').textContent = state.seat === 'host' ? 'End chat' : 'Leave chat';
  if (state.inviteUrl) {
    hide($('#invite-bar'), false);
    $('#invite-url').value = state.inviteUrl;
  } else {
    hide($('#invite-bar'), true);
  }
  history.replaceState(null, '', roomPath(state.roomId));
  updateConnectionStatus('Connecting…', false);
}
function processServer(event) {
  let payload;
  try { payload = JSON.parse(event.data); } catch { return; }
  switch (payload.type) {
    case 'joined':
      state.seat = payload.seat;
      $('#leave-room').textContent = state.seat === 'host' ? 'End chat' : 'Leave chat';
      updateConnectionStatus('Waiting for your friend…', false);
      notice('', false);
      break;
    case 'presence':
      if (payload.online) {
        updateConnectionStatus((payload.name || 'Your friend') + ' is here · Encrypted in transit', true);
      } else {
        updateConnectionStatus('Your friend is away · Their link can reconnect', false);
        text($('#typing'), '');
      }
      break;
    case 'message':
      appendBubble('theirs', payload.text, payload.name);
      text($('#typing'), '');
      break;
    case 'sent':
      if (state.pending.length) appendBubble('mine', state.pending.shift(), state.name);
      break;
    case 'typing':
      text($('#typing'), payload.active ? 'Your friend is typing…' : '');
      break;
    case 'error':
      state.pending.shift();
      notice(payload.message || 'Message could not be sent.');
      break;
    case 'ended':
      notice('This conversation has ended. Messages were not saved.', false);
      updateConnectionStatus('Room closed', false);
      clearRoom();
      break;
    default:
      break;
  }
}
function connect() {
  if (!state.roomId || !state.token || state.intentionalClose) return;
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const address = protocol + '//' + location.host + '/ws/' + encodeURIComponent(state.roomId);
  const socket = new WebSocket(address);
  state.socket = socket;
  socket.addEventListener('open', () => {
    state.reconnectAttempts = 0;
    socket.send(JSON.stringify({type:'join', token:state.token, name:state.name}));
  });
  socket.addEventListener('message', processServer);
  socket.addEventListener('close', () => {
    if (state.socket !== socket || state.intentionalClose) return;
    updateConnectionStatus('Disconnected · Reconnecting…', false);
    state.pending.length = 0;
    const delay = Math.min(1000 * (2 ** state.reconnectAttempts), 12000);
    state.reconnectAttempts += 1;
    window.setTimeout(connect, delay);
  });
  socket.addEventListener('error', () => updateConnectionStatus('Connection lost · Retrying…', false));
}
async function createRoom() {
  const button = $('#create-room'); button.disabled = true;
  notice('');
  try {
    const room = await requestJSON('/api/rooms', {method:'POST'});
    state.roomId = room.room_id;
    state.token = room.host_key;
    state.inviteUrl = room.invite_url;
    state.seat = 'host';
    state.name = $('#nickname').value.trim().slice(0,24) || 'You';
    store(state.roomId, {token:state.token, seat:'host', name:state.name, inviteUrl:state.inviteUrl});
    showChat(); connect();
  } catch (error) { notice(error.message); }
  finally { button.disabled = false; }
}
async function joinRoom() {
  const button = $('#join-room'); button.disabled = true;
  notice('');
  try {
    await requestJSON('/api/rooms/' + encodeURIComponent(state.roomId));
    state.name = $('#join-name').value.trim().slice(0,24) || 'Friend';
    state.seat = 'guest';
    store(state.roomId, {token:state.token, seat:'guest', name:state.name, inviteUrl:''});
    showChat(); connect();
  } catch (error) { notice(error.message); button.disabled = false; }
}
async function initialize() {
  const match = location.pathname.match(/^\/r\/([a-f0-9]{14})$/);
  if (!match) return;
  const roomId = match[1];
  const link = new URL(location.href);
  const invite = new URLSearchParams(link.hash.replace(/^#/, '')).get('invite');
  const saved = stored(roomId);
  state.roomId = roomId;
  if (invite && invite.length >= 24) {
    state.token = invite;
    history.replaceState(null, '', roomPath(roomId));
    hide($('#hero')); hide($('#join-panel'), false); hide($('#reassurance'));
  } else if (saved?.token) {
    state.token = saved.token;
    state.name = saved.name || 'Friend';
    state.seat = saved.seat;
    state.inviteUrl = saved.inviteUrl || '';
    showChat(); connect();
  } else {
    hide($('#hero')); hide($('#join-panel'), false); hide($('#reassurance'));
    notice('This room needs a valid invitation link. Ask your friend for a new link.', true);
    $('#join-room').disabled = true;
  }
}
function send() {
  if (!state.friendOnline || !state.socket || state.socket.readyState !== WebSocket.OPEN) return;
  const content = $('#message').value.trim();
  if (!content || content.length > 1000) return;
  state.pending.push(content);
  state.socket.send(JSON.stringify({type:'message', text:content}));
  $('#message').value = '';
}
$('#create-room').addEventListener('click', createRoom);
$('#join-room').addEventListener('click', joinRoom);
$('#copy-link').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText($('#invite-url').value);
    text($('#copy-link'), 'Copied ✓');
    setTimeout(() => text($('#copy-link'), 'Copy link'), 1800);
  } catch {
    $('#invite-url').focus();
    $('#invite-url').select();
    notice('Select and copy the invitation link manually.', false);
  }
});
$('#composer').addEventListener('submit', (event) => { event.preventDefault(); send(); });
$('#message').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault(); send();
  }
});
$('#message').addEventListener('input', () => {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;
  state.socket.send(JSON.stringify({type:'typing', active:true}));
  clearTimeout(state.typingTimer);
  state.typingTimer = setTimeout(() => {
    if (state.socket?.readyState === WebSocket.OPEN) state.socket.send(JSON.stringify({type:'typing', active:false}));
  }, 900);
});
$('#leave-room').addEventListener('click', () => {
  if (!confirm(state.seat === 'host'
      ? 'End this chat for both people? Unsaved messages will disappear.'
      : 'Leave this conversation? Unsaved messages will disappear.')) return;
  if (state.socket?.readyState === WebSocket.OPEN) state.socket.send(JSON.stringify({type:state.seat === 'host'?'end':'leave'}));
  clearRoom();
  location.assign('/');
});
initialize();
