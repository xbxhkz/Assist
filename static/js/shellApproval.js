// Polls /api/shell/pending for the active chat session and renders an approval
// card for each state-changing command: Approve / Deny / Auto-approve all.
//
// Session id: this front-end tracks the active chat session via the ES-module
// `sessions.js`, which app.js publishes as `window.sessionModule` (see
// static/app.js, `window.sessionModule = sessionModule;`, evaluated at module
// load — before DOMContentLoaded fires, since <script type=module> runs like
// `defer`). `window.sessionModule.getCurrentSessionId()` is the same accessor
// operator.js/chat.js use everywhere else in this file tree.
(function () {
  const POLL_MS = 1500;

  function sid() {
    return (window.sessionModule && window.sessionModule.getCurrentSessionId
      && window.sessionModule.getCurrentSessionId()) || '';
  }

  async function decide(pending_id, decision) {
    try {
      await fetch('/api/shell/decision', {
        method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid(), pending_id, decision }),
      });
    } catch (e) { /* next poll re-renders */ }
    poll();
  }

  function btn(label, fn) {
    const b = document.createElement('button');
    b.type = 'button'; b.textContent = label; b.addEventListener('click', fn);
    return b;
  }

  function render(list) {
    const host = document.getElementById('shell-approval-host');
    if (!host) return;
    const sig = JSON.stringify(list.map((p) => p.pending_id));
    if (sig === host.dataset.sig) return;      // avoid wiping focus each poll
    host.dataset.sig = sig;
    while (host.firstChild) host.removeChild(host.firstChild);
    list.forEach((p) => {
      const card = document.createElement('div');
      card.className = 'list-item';
      card.style.cssText = 'display:block;padding:8px;border:1px solid var(--yellow,#f1fa8c);border-radius:6px;margin:6px 0;';
      const h = document.createElement('div');
      h.style.cssText = 'font-weight:600;font-size:12px;margin-bottom:4px;';
      h.textContent = 'Run this ' + p.shell + ' command?';
      const pre = document.createElement('pre');
      pre.style.cssText = 'font-size:12px;white-space:pre-wrap;word-break:break-word;margin:4px 0;';
      pre.textContent = p.command;
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;gap:6px;';
      row.appendChild(btn('Approve', () => decide(p.pending_id, 'approve')));
      row.appendChild(btn('Deny', () => decide(p.pending_id, 'deny')));
      row.appendChild(btn('Auto-approve all', () => decide(p.pending_id, 'auto_approve_all')));
      card.appendChild(h); card.appendChild(pre); card.appendChild(row);
      host.appendChild(card);
    });
  }

  async function poll() {
    const id = sid();
    if (!id) { render([]); return; }        // no active session yet — nothing to show
    try {
      const res = await fetch('/api/shell/pending?session_id=' + encodeURIComponent(id), { credentials: 'same-origin' });
      if (res.ok) render((await res.json()).pending || []);
    } catch (e) { /* ignore; keep polling */ }
  }

  function ensureHost() {
    let host = document.getElementById('shell-approval-host');
    if (host) return host;
    // Static placeholder lives in index.html between #chat-history and
    // #attach-strip so approval cards stay visible above the composer and
    // survive chat-history re-renders (that container's innerHTML is
    // rewritten on session switch/history load). This is a defensive
    // fallback only, in case the placeholder is ever removed from the DOM.
    host = document.createElement('div');
    host.id = 'shell-approval-host';
    const chatHistory = document.getElementById('chat-history');
    if (chatHistory && chatHistory.parentNode) {
      chatHistory.parentNode.insertBefore(host, chatHistory.nextSibling);
    } else {
      (document.getElementById('chat-container') || document.body).appendChild(host);
    }
    return host;
  }

  document.addEventListener('DOMContentLoaded', () => {
    ensureHost();
    setInterval(poll, POLL_MS);
    poll();
  });
})();
