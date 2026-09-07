const syncChannel = new BroadcastChannel('avam_state_channel');
let activeDocId = null;
let activeDocName = '';

document.addEventListener('DOMContentLoaded', () => {
  const params = new URLSearchParams(window.location.search);
  const docParam = params.get('doc');
  if (docParam) {
    activeDocName = docParam;
    document.getElementById('pdf-viewer').src = '/uploads/' + encodeURIComponent(docParam);
  }

  const stored = localStorage.getItem('avam_active_document');
  if (stored) {
    const data = JSON.parse(stored);
    if (!docParam) {
      activeDocName = data.name;
      document.getElementById('pdf-viewer').src = data.url;
    }
  }

  fetchDocumentHistory();
  pollModelStatus();

  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const pane = document.getElementById('tab-' + btn.dataset.tab);
      if (pane) pane.classList.add('active');
    });
  });

  document.getElementById('pdf-upload').addEventListener('change', handleUpload);
  document.getElementById('doc-history').addEventListener('change', e => {
    if (e.target.value) {
      const opt = e.target.options[e.target.selectedIndex];
      setActiveDoc(e.target.value, opt.textContent);
    }
  });
  document.getElementById('chat-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') sendMessage();
  });

  window.addEventListener('storage', event => {
    if (event.key === 'avam_active_document' && event.newValue) {
      try {
        const data = JSON.parse(event.newValue);
        setActiveDoc(data.url, data.name, false);
      } catch (_) {}
    }
  });
  syncChannel.onmessage = msg => {
    if (msg.data && msg.data.type === 'DOCUMENT_SWITCH') {
      setActiveDoc(msg.data.payload.url, msg.data.payload.name, false);
    }
  };
});

function pollModelStatus() {
  const badge = document.getElementById('model-status');
  if (!badge) return;
  const check = () => {
    fetch('/api/status')
      .then(r => r.json())
      .then(data => {
        if (data.ready) {
          badge.textContent = '\u2713 All models ready';
          badge.className = 'status-badge ready';
        } else {
          const loaded = Object.values(data.models).filter(s => s === 'ready').length;
          const total = Object.keys(data.models).length;
          badge.textContent = 'Loading models... (' + loaded + '/' + total + ')';
          badge.className = 'status-badge loading';
          setTimeout(check, 3000);
        }
      })
      .catch(() => setTimeout(check, 3000));
  };
  check();
}

function setActiveDoc(url, name, broadcast) {
  if (broadcast === undefined) broadcast = true;
  activeDocName = name;
  const iframe = document.getElementById('pdf-viewer');
  if (iframe) iframe.src = url;

  const payload = { url: url, name: name, timestamp: Date.now() };
  try {
    localStorage.setItem('avam_active_document', JSON.stringify(payload));
  } catch (_) {}

  const curUrl = new URL(window.location.href);
  curUrl.searchParams.set('doc', name);
  window.history.pushState({ doc: name }, '', curUrl.toString());

  if (broadcast) {
    try {
      syncChannel.postMessage({ type: 'DOCUMENT_SWITCH', payload: payload });
    } catch (_) {}
  }
}

async function fetchDocumentHistory() {
  try {
    const res = await fetch('/api/documents/history');
    const data = await res.json();
    const sel = document.getElementById('doc-history');
    if (!sel) return;
    sel.innerHTML = '<option value="">Select document...</option>';
    if (data.documents) {
      data.documents.forEach(function(doc) {
        const opt = document.createElement('option');
        opt.value = doc.url;
        opt.textContent = doc.name;
        sel.appendChild(opt);
      });
    }
  } catch (_) {}
}

async function handleUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append('file', file);
  try {
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    if (res.ok) {
      const data = await res.json();
      await fetchDocumentHistory();
      setActiveDoc(data.pdf_url, data.filename, true);
    }
  } catch (e) { console.error(e); }
}

async function sendMessage() {
  const input = document.getElementById('chat-input');
  if (!input) return;
  const q = input.value.trim();
  if (!q || !activeDocName) return;
  input.value = '';
  const sendBtn = input.nextElementSibling;
  if (sendBtn) sendBtn.disabled = true;

  const box = document.getElementById('chat-messages');
  if (!box) return;
  const emptyState = box.querySelector('.empty-state');
  if (emptyState) emptyState.remove();

  const userMsg = document.createElement('div');
  userMsg.className = 'chat-msg user';
  userMsg.textContent = q;
  box.appendChild(userMsg);
  box.scrollTop = box.scrollHeight;

  const skel = document.createElement('div');
  skel.className = 'skeleton-card';
  skel.innerHTML = '<div class="skeleton-bar title"></div><div class="skeleton-bar"></div><div class="skeleton-bar short"></div>';
  box.appendChild(skel);

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q })
    });
    const data = await res.json();
    skel.remove();

    if (data.error) {
      const err = document.createElement('div');
      err.className = 'chat-msg assistant';
      err.textContent = 'Error: ' + data.error;
      box.appendChild(err);
      if (sendBtn) sendBtn.disabled = false;
      return;
    }

    const ans = document.createElement('div');
    ans.className = 'chat-msg assistant';
    ans.innerHTML = '<div>' + escapeHtml(data.answer || 'No answer generated.') + '</div>';

    if (data.grounding && data.grounding.warning) {
      ans.innerHTML += '<div class="grounding-warning">\u26A0 ' + escapeHtml(data.grounding.warning) + '</div>';
    }

    const actions = document.createElement('div');
    actions.className = 'msg-actions';
    const answerText = data.answer || '';
    actions.innerHTML =
      '<button onclick="copyMsg(this)">\uD83D\uDCCB Copy</button>' +
      '<button onclick="regenerateMsg(this)">\uD83D\uDD04 Regenerate</button>' +
      feedbackWidgetHTML('qa', answerText);
    ans.appendChild(actions);
    box.appendChild(ans);
    box.scrollTop = box.scrollHeight;
  } catch (e) {
    skel.remove();
    const err = document.createElement('div');
    err.className = 'chat-msg assistant';
    err.textContent = 'Network error: ' + e.message;
    box.appendChild(err);
  }
  if (sendBtn) sendBtn.disabled = false;
}

function escapeHtml(str) {
  var div = document.createElement('div');
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}

function copyMsg(btn) {
  var msg = btn.closest('.chat-msg');
  if (!msg) return;
  var textNode = msg.childNodes[0];
  if (!textNode) return;
  navigator.clipboard.writeText(textNode.textContent || '');
  btn.textContent = '\u2713 Copied';
  setTimeout(function() { btn.textContent = '\uD83D\uDCCB Copy'; }, 2000);
}

async function regenerateMsg(btn) {
  var msg = btn.closest('.chat-msg');
  if (!msg) return;
  var prev = msg.previousElementSibling;
  var question = '';
  if (prev && prev.classList.contains('chat-msg') && prev.classList.contains('user')) {
    question = prev.textContent || '';
  }
  msg.remove();
  var input = document.getElementById('chat-input');
  if (input) {
    input.value = question;
    sendMessage();
  }
}

async function generateQuiz() {
  var container = document.getElementById('quiz-content');
  if (!container) return;
  if (!activeDocName) {
    container.innerHTML = '<p class="empty-state">Please select a document first.</p>';
    return;
  }
  container.innerHTML = '<div class="skeleton-card"><div class="skeleton-bar title"></div><div class="skeleton-bar"></div><div class="skeleton-bar"></div><div class="skeleton-bar short"></div></div>';

  try {
    var res = await fetch('/api/generate/quiz', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
    var data = await res.json();
    if (data.error) { container.innerHTML = '<p class="empty-state">' + escapeHtml(data.error) + '</p>'; return; }

    var html = '<h3 style="margin-bottom:16px;">' + escapeHtml(data.quiz_title || 'Quiz') + '</h3>';
    if (data.questions) {
      data.questions.forEach(function(q, i) {
        html += '<div class="quiz-item">';
        html += '<h4>' + (i + 1) + '. ' + escapeHtml(q.question_text) + '</h4>';
        if (q.choices) {
          q.choices.forEach(function(c, ci) {
            var correctIdx = q.choices.indexOf(q.answer);
            html += '<div class="quiz-option" onclick="checkQuizAnswer(this,' + ci + ',' + correctIdx + ')">' + escapeHtml(c) + '</div>';
          });
        }
        html += '<div class="quiz-explanation" style="display:none;">';
        html += '<strong>Answer:</strong> ' + escapeHtml(q.answer) + '<br>';
        html += escapeHtml(q.explanation || '');
        html += '</div></div>';
      });
    }
    container.innerHTML = html;
  } catch (e) {
    container.innerHTML = '<p class="empty-state">Error: ' + escapeHtml(e.message) + '</p>';
  }
}

function checkQuizAnswer(el, selected, correctIdx) {
  var item = el.closest('.quiz-item');
  if (!item) return;
  var opts = item.querySelectorAll('.quiz-option');
  opts.forEach(function(o, i) {
    o.style.pointerEvents = 'none';
    if (i === correctIdx) o.classList.add('correct');
    else if (i === selected && i !== correctIdx) o.classList.add('wrong');
  });
  var exp = item.querySelector('.quiz-explanation');
  if (exp) exp.style.display = 'block';
}

async function generateFlashcards() {
  var container = document.getElementById('flashcard-content');
  if (!container) return;
  if (!activeDocName) {
    container.innerHTML = '<p class="empty-state">Please select a document first.</p>';
    return;
  }
  container.innerHTML = '<div class="skeleton-card"><div class="skeleton-bar title"></div><div class="skeleton-bar"></div><div class="skeleton-bar"></div><div class="skeleton-bar short"></div></div>';

  try {
    var res = await fetch('/api/generate/flashcards', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
    var data = await res.json();
    if (data.error) { container.innerHTML = '<p class="empty-state">' + escapeHtml(data.error) + '</p>'; return; }

    var html = '<h3 style="margin-bottom:16px;">' + escapeHtml(data.deck_name || 'Flashcards') + '</h3>';
    if (data.cards) {
      data.cards.forEach(function(card) {
        html += '<div class="flashcard-item" onclick="var b=this.querySelector(\'.card-back\');if(b)b.style.display=b.style.display===\'none\'?\'block\':\'none\';">';
        html += '<strong>Q:</strong> ' + escapeHtml(card.front);
        html += '<div class="card-back" style="display:none;"><strong>A:</strong> ' + escapeHtml(card.back) + '</div>';
        html += '</div>';
      });
    }
    if (data.cards && data.cards.length > 0) {
      var dataStr = JSON.stringify(data).replace(/'/g, "\\'");
      html += '<button onclick="exportAnki(' + "'" + dataStr.replace(/"/g, '&quot;') + "'" + ')" class="btn-primary" style="margin-top:16px;">\uD83D\uDCE6 Export to Anki</button>';
    }
    container.innerHTML = html;
  } catch (e) {
    container.innerHTML = '<p class="empty-state">Error: ' + escapeHtml(e.message) + '</p>';
  }
}

async function exportAnki(dataStr) {
  try {
    var data = typeof dataStr === 'string' ? JSON.parse(dataStr) : dataStr;
    var res = await fetch('/api/export/anki', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cards: data.cards, title: data.deck_name || 'Avam Deck' })
    });
    if (res.ok) {
      var blob = await res.blob();
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = (data.deck_name || 'deck').replace(/\s+/g, '_').toLowerCase() + '.apkg';
      document.body.appendChild(a);
      a.click();
      setTimeout(function() { document.body.removeChild(a); URL.revokeObjectURL(url); }, 100);
    }
  } catch (e) { alert('Export failed: ' + e.message); }
}

async function generateGraph() {
  var container = document.getElementById('graph-container');
  if (!container) return;
  if (!activeDocName) {
    container.innerHTML = '<p class="empty-state">Select a document first.</p>';
    return;
  }
  container.innerHTML = '<div class="skeleton-card"><div class="skeleton-bar title"></div><div class="skeleton-bar"></div><div class="skeleton-bar short"></div></div>';

  try {
    var res = await fetch('/api/generate/graph', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
    var data = await res.json();
    container.innerHTML = '';

    if (data.nodes && data.edges && window.vis) {
      var nodes = new vis.DataSet(data.nodes.map(function(n) {
        return { id: n.id, label: n.label, color: n.color || '#3b82f6', title: 'Type: ' + n.type };
      }));
      var edges = new vis.DataSet(data.edges.map(function(e) {
        return { from: e.source, to: e.target, label: e.label || '' };
      }));
      new vis.Network(container, { nodes: nodes, edges: edges }, {
        physics: { solver: 'forceAtlas2Based', forceAtlas2Based: { gravitationalConstant: -40 } },
        interaction: { navigationButtons: true, zoomView: true, dragView: true },
        nodes: { shape: 'dot', size: 16, font: { size: 14, color: '#fffffe' } },
        edges: { font: { size: 10, color: '#a7a9be' }, arrows: { to: { enabled: true, scaleFactor: 0.5 } } }
      });
    } else if (data.error) {
      container.innerHTML = '<p class="empty-state">' + escapeHtml(data.error) + '</p>';
    } else {
      container.innerHTML = '<p class="empty-state">No entities found in document.</p>';
    }
  } catch (e) {
    container.innerHTML = '<p class="empty-state">Error: ' + escapeHtml(e.message) + '</p>';
  }
}

function feedbackWidgetHTML(feature, outputText) {
  var escaped = (outputText || '').replace(/'/g, "\\'");
  return '<div class="feedback-container">' +
    '<button class="feedback-btn" onclick="toggleFeedback(this)">\uD83D\uDC4E Report</button>' +
    '<div class="feedback-panel" style="display:none;position:absolute;bottom:100%;right:0;width:280px;z-index:100;">' +
    '<p style="margin-bottom:8px;color:var(--text-dim);">Anonymous feedback — no account needed, no IP logged.</p>' +
    '<select id="fb-rating-' + feature + '" onchange=""><option value="">What went wrong?</option>' +
    '<option value="hallucinated">Looks made up / not from my document</option>' +
    '<option value="off_topic">Answered the wrong question</option>' +
    '<option value="bad">Not useful or too vague</option>' +
    '<option value="privacy">Privacy concern</option>' +
    '<option value="great">Actually great — misclicked</option></select>' +
    '<textarea id="fb-desc-' + feature + '" placeholder="What did you expect?" rows="2"></textarea>' +
    '<label><input type="checkbox" id="fb-include-' + feature + '"> Include AI response</label>' +
    '<button onclick="submitFeedback(\'' + feature + '\',\'' + escaped + '\')" class="btn-primary" style="margin-top:8px;width:100%;padding:8px;">Submit</button>' +
    '</div></div>';
}

function toggleFeedback(btn) {
  var panel = btn.parentElement.querySelector('.feedback-panel');
  if (panel) panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
}

async function submitFeedback(feature, outputText) {
  var container = event.target.closest('.feedback-container');
  if (!container) return;
  var rating = container.querySelector('#fb-rating-' + feature);
  var desc = container.querySelector('#fb-desc-' + feature);
  var include = container.querySelector('#fb-include-' + feature);
  if (!rating || !rating.value) return;
  try {
    var res = await fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        feature: feature,
        rating: rating.value,
        description: desc ? desc.value : '',
        include_output: include ? include.checked : false,
        output_excerpt: include && include.checked ? (outputText || '').slice(0, 500) : ''
      })
    });
    var data = await res.json();
    var panel = container.querySelector('.feedback-panel');
    if (panel) panel.innerHTML = '<p>\u2713 Feedback noted (ref: ' + (data.reference_code || '') + ')</p>';
  } catch (e) { alert('Feedback failed'); }
}

function toggleTheme() {
  var root = document.documentElement;
  var cur = root.getAttribute('data-theme');
  root.setAttribute('data-theme', cur === 'light' ? 'dark' : 'light');
}

async function deleteDocument() {
  if (!activeDocName) return;
  if (!confirm('Delete "' + activeDocName + '"?')) return;
  try {
    var res = await fetch('/document/' + encodeURIComponent(activeDocName), { method: 'DELETE' });
    var data = await res.json();
    if (data.success) {
      activeDocName = '';
      document.getElementById('pdf-viewer').src = 'about:blank';
      document.getElementById('chat-messages').innerHTML = '<div class="empty-state">Ask a question about your document...</div>';
      document.getElementById('doc-history').value = '';
      fetchDocumentHistory();
    } else {
      alert('Delete failed: ' + (data.error || 'unknown'));
    }
  } catch (e) { alert('Delete error: ' + e.message); }
}

async function generateSummary() {
  var container = document.getElementById('summary-content');
  if (!container) return;
  if (!activeDocName) { container.innerHTML = '<p class="empty-state">Please select a document first.</p>'; return; }
  container.innerHTML = '<div class="skeleton-card"><div class="skeleton-bar title"></div><div class="skeleton-bar"></div><div class="skeleton-bar"></div><div class="skeleton-bar short"></div></div>';
  var detail = document.getElementById('summary-detail') ? document.getElementById('summary-detail').value : 'standard';
  var focus = document.getElementById('summary-focus') ? document.getElementById('summary-focus').value : '';
  try {
    var res = await fetch('/summary', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ doc_id: activeDocName, detail: detail, focus: focus }) });
    var data = await res.json();
    if (data.error) { container.innerHTML = '<p class="empty-state">' + escapeHtml(data.error) + '</p>'; return; }
    var html = '';
    if (data.stats) html += '<div style="color:var(--text-dim);font-size:13px;margin-bottom:12px;">Model: ' + escapeHtml(data.stats.model_used || '') + ' &middot; Compression: ' + escapeHtml(data.stats.compression || '') + '</div>';
    html += '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px;margin-bottom:16px;line-height:1.7;">' + escapeHtml(data.overall || '') + '</div>';
    if (data.key_concepts && data.key_concepts.length) {
      html += '<h4 style="margin-bottom:8px;">Key Concepts</h4><div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;">';
      data.key_concepts.forEach(function(c) { html += '<span style="background:var(--accent);color:white;padding:4px 12px;border-radius:20px;font-size:13px;">' + escapeHtml(c) + '</span>'; });
      html += '</div>';
    }
    if (data.sections && data.sections.length > 1) {
      data.sections.forEach(function(s) {
        if (s.summary) html += '<h4 style="margin:12px 0 6px;">' + escapeHtml(s.title) + '</h4><p style="color:var(--text-muted);line-height:1.6;">' + escapeHtml(s.summary) + '</p>';
      });
    }
    container.innerHTML = html;
  } catch (e) { container.innerHTML = '<p class="empty-state">Error: ' + escapeHtml(e.message) + '</p>'; }
}

async function previewSlides() {
  var container = document.getElementById('slides-content');
  if (!container) return;
  if (!activeDocName) { container.innerHTML = '<p class="empty-state">Please select a document first.</p>'; return; }
  container.innerHTML = '<div class="skeleton-card"><div class="skeleton-bar title"></div><div class="skeleton-bar"></div><div class="skeleton-bar"></div></div>';
  var detail = document.getElementById('slides-detail') ? document.getElementById('slides-detail').value : 'standard';
  var count = document.getElementById('slides-count') ? document.getElementById('slides-count').value : null;
  var focus = document.getElementById('slides-focus') ? document.getElementById('slides-focus').value : '';
  try {
    var res = await fetch('/pptx/preview', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ doc_id: activeDocName, detail: detail, slide_count: count ? parseInt(count) : null, focus: focus }) });
    var data = await res.json();
    if (data.error) { container.innerHTML = '<p class="empty-state">' + escapeHtml(data.error) + '</p>'; return; }
    var html = '<p style="color:var(--text-dim);margin-bottom:12px;">' + (data.count || 0) + ' slides</p>';
    if (data.slides) {
      data.slides.forEach(function(s, i) {
        html += '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px;margin-bottom:8px;">';
        html += '<span style="font-size:12px;color:var(--text-dim);">Slide ' + (i + 1) + '</span>';
        html += '<h4 style="margin-top:4px;">' + escapeHtml(s.title || '') + '</h4>';
        if (s.content) html += '<p style="color:var(--text-muted);font-size:13px;margin-top:6px;">' + escapeHtml(s.content.slice(0, 200)) + '</p>';
        html += '</div>';
      });
    }
    container.innerHTML = html;
  } catch (e) { container.innerHTML = '<p class="empty-state">Error: ' + escapeHtml(e.message) + '</p>'; }
}

async function downloadSlides() {
  if (!activeDocName) { alert('Select a document first.'); return; }
  var detail = document.getElementById('slides-detail') ? document.getElementById('slides-detail').value : 'standard';
  var count = document.getElementById('slides-count') ? document.getElementById('slides-count').value : null;
  var focus = document.getElementById('slides-focus') ? document.getElementById('slides-focus').value : '';
  try {
    var res = await fetch('/pptx', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ doc_id: activeDocName, title: activeDocName, detail: detail, slide_count: count ? parseInt(count) : null, focus: focus }) });
    if (res.headers.get('content-type') && res.headers.get('content-type').includes('json')) {
      var err = await res.json();
      alert('Error: ' + (err.error || 'unknown'));
      return;
    }
    var blob = await res.blob();
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = (activeDocName || 'presentation').replace(/\.[^.]+$/, '') + '_slides.pptx';
    document.body.appendChild(a);
    a.click();
    setTimeout(function() { document.body.removeChild(a); URL.revokeObjectURL(url); }, 100);
  } catch (e) { alert('Download failed: ' + e.message); }
}

async function transcribeAudio() {
  var container = document.getElementById('audio-content');
  var progress = document.getElementById('audio-progress');
  if (!container) return;
  var fileInput = document.getElementById('audio-file');
  var urlInput = document.getElementById('audio-url');
  var file = fileInput ? fileInput.files[0] : null;
  var url = urlInput ? urlInput.value.trim() : '';
  if (!file && !url) { container.innerHTML = '<p class="empty-state">Select an audio file or paste a YouTube URL.</p>'; return; }
  container.innerHTML = '<div class="skeleton-card"><div class="skeleton-bar title"></div><div class="skeleton-bar"></div><div class="skeleton-bar short"></div></div>';
  if (progress) progress.textContent = 'Transcribing... this may take a while.';
  var fd = new FormData();
  if (file) fd.append('file', file);
  if (url) fd.append('url', url);
  try {
    var res = await fetch('/transcribe', { method: 'POST', body: fd });
    var data = await res.json();
    if (data.success) {
      container.innerHTML = '<div style="text-align:center;padding:40px 20px;"><span style="font-size:48px;">&#x2705;</span><h3 style="margin:12px 0;">Transcription complete</h3><p style="color:var(--text-muted);">' + escapeHtml(data.document.name) + ' (' + data.document.word_count + ' words)</p></div>';
      if (progress) progress.textContent = '';
      fetchDocumentHistory();
    } else {
      container.innerHTML = '<p class="empty-state">Error: ' + escapeHtml(data.error || 'Transcription failed') + '</p>';
      if (progress) progress.textContent = '';
    }
  } catch (e) { container.innerHTML = '<p class="empty-state">Error: ' + escapeHtml(e.message) + '</p>'; if (progress) progress.textContent = ''; }
}

async function generateAudioOverview() {
  var container = document.getElementById('audio-overview-content');
  if (!container) return;
  if (!activeDocName) { container.innerHTML = '<p class="empty-state">Please select a document first.</p>'; return; }
  container.innerHTML = '<div class="skeleton-card"><div class="skeleton-bar title"></div><div class="skeleton-bar"></div><div class="skeleton-bar"></div></div>';
  try {
    var res = await fetch('/api/generate/audio-overview', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
    var data = await res.json();
    if (data.error) { container.innerHTML = '<p class="empty-state">' + escapeHtml(data.error) + '</p>'; return; }
    var html = '<div style="margin-bottom:12px;">';
    if (data.audio_url) {
      html += '<audio controls style="width:100%;margin-bottom:12px;"><source src="' + data.audio_url + '" type="audio/mpeg"></audio>';
    }
    if (data.script) {
      html += '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px;max-height:400px;overflow-y:auto;line-height:1.7;font-size:14px;">';
      data.script.split('\n').forEach(function(line) {
        if (line.startsWith('Host1:')) {
          html += '<p style="margin-bottom:6px;"><strong style="color:var(--accent);">' + escapeHtml(line) + '</strong></p>';
        } else if (line.startsWith('Host2:')) {
          html += '<p style="margin-bottom:6px;padding-left:16px;"><strong style="color:#22c55e;">' + escapeHtml(line) + '</strong></p>';
        } else {
          html += '<p style="margin-bottom:6px;">' + escapeHtml(line) + '</p>';
        }
      });
      html += '</div>';
    }
    html += '</div>';
    container.innerHTML = html;
  } catch (e) { container.innerHTML = '<p class="empty-state">Error: ' + escapeHtml(e.message) + '</p>'; }
}

async function generateVideo() {
  var container = document.getElementById('video-content');
  if (!container) return;
  if (!activeDocName) { container.innerHTML = '<p class="empty-state">Please select a document first.</p>'; return; }
  container.innerHTML = '<div style="text-align:center;padding:40px;"><div class="skeleton-bar" style="width:300px;margin:0 auto 16px;"></div><div class="skeleton-bar" style="width:200px;margin:0 auto;"></div><p style="color:var(--text-dim);margin-top:16px;">Generating video (slides + narration)... this may take a minute.</p></div>';
  try {
    var res = await fetch('/api/generate/video', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
    var data = await res.json();
    if (data.error) { container.innerHTML = '<p class="empty-state">' + escapeHtml(data.error) + '</p>'; return; }
    var html = '<div style="margin-bottom:12px;">';
    if (data.video_url) {
      html += '<video controls style="width:100%;border-radius:var(--radius);background:#000;">';
      html += '<source src="' + data.video_url + '" type="video/mp4">';
      html += '</video>';
    }
    html += '<p style="color:var(--text-dim);font-size:13px;margin-top:8px;">' + data.slide_count + ' slides &middot; ' + escapeHtml(data.document_name) + '</p>';
    html += '</div>';
    container.innerHTML = html;
  } catch (e) { container.innerHTML = '<p class="empty-state">Error: ' + escapeHtml(e.message) + '</p>'; }
}
