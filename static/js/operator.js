// AI Operator panel: goal box + live transcript of proposed actions with a
// human-in-the-loop approval flow (Approve/Deny/Edit/Stop). Mirrors the plain
// IIFE, CSP-safe style of plugins.js (createElement + addEventListener only —
// no inline handlers).
(function () {
  function $(id) { return document.getElementById(id); }

  const ACTIVE_STATUSES = ['running', 'starting', 'awaiting_confirmation', 'awaiting_answer'];
  const TERMINAL_STATUSES = ['idle', 'done', 'stopped', 'round_cap', 'time_cap', 'stuck', 'error'];
  const POLL_MS = 1500;

  let pollTimer = null;

  function open() {
    const m = $('operator-modal');
    if (m) m.classList.remove('hidden');
    poll();
  }

  function close() {
    const m = $('operator-modal');
    if (m) m.classList.add('hidden');
  }

  function setMsg(text, isError) {
    const el = $('operator-msg');
    if (!el) return;
    el.textContent = text || '';
    el.style.color = isError ? 'var(--red, #ff5555)' : '';
  }

  function setActive(active) {
    const indicator = $('operator-active-indicator');
    if (indicator) indicator.style.display = active ? '' : 'none';
    const stopBtn = $('operator-stop-btn');
    if (stopBtn) stopBtn.style.display = active ? '' : 'none';
    const startBtn = $('operator-start-btn');
    if (startBtn) startBtn.disabled = !!active;
  }

  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  function startPolling() {
    stopPolling();
    pollTimer = setInterval(poll, POLL_MS);
  }

  async function start() {
    const goalInp = $('operator-goal');
    const goal = goalInp ? goalInp.value.trim() : '';
    if (!goal) { setMsg('Enter a goal first', true); return; }
    const startBtn = $('operator-start-btn');
    if (startBtn) startBtn.disabled = true;
    setMsg('Starting…', false);
    try {
      const res = await fetch('/api/operator/start', {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goal }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error((data && data.detail) || String(res.status));
      startPolling();
      await poll();
    } catch (e) {
      setMsg(`Failed to start: ${e.message}`, true);
      if (startBtn) startBtn.disabled = false;
    }
  }

  async function stop() {
    try {
      const res = await fetch('/api/operator/stop', { method: 'POST', credentials: 'same-origin' });
      if (!res.ok) throw new Error(String(res.status));
      await poll();
    } catch (e) {
      setMsg(`Failed to stop: ${e.message}`, true);
    }
  }

  async function sendDecision(payload) {
    try {
      const res = await fetch('/api/operator/decision', {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error((data && data.detail) || String(res.status));
      await poll();
    } catch (e) {
      setMsg(`Decision failed: ${e.message}`, true);
    }
  }

  function renderTranscriptEntry(entry) {
    const row = document.createElement('div');
    row.className = 'list-item';
    row.style.cssText = 'display:block;padding:6px 8px;';

    const tool = document.createElement('div');
    tool.style.cssText = 'font-weight:600;font-size:12px;';
    tool.textContent = entry.tool || '';
    row.appendChild(tool);

    if (entry.rationale) {
      const rationale = document.createElement('div');
      rationale.style.cssText = 'font-size:11px;opacity:0.75;';
      rationale.textContent = entry.rationale;
      row.appendChild(rationale);
    }

    if (entry.args && Object.keys(entry.args).length) {
      const args = document.createElement('div');
      args.style.cssText = 'font-size:11px;opacity:0.6;white-space:pre-wrap;word-break:break-word;';
      args.textContent = JSON.stringify(entry.args);
      row.appendChild(args);
    }

    if (entry.obs !== undefined && entry.obs !== null) {
      const obs = document.createElement('div');
      obs.style.cssText = 'font-size:11px;opacity:0.5;white-space:pre-wrap;word-break:break-word;';
      obs.textContent = `→ ${typeof entry.obs === 'string' ? entry.obs : JSON.stringify(entry.obs)}`;
      row.appendChild(obs);
    }

    return row;
  }

  function actionCardShell(titleText) {
    const card = document.createElement('div');
    card.id = 'operator-pending-card';
    card.style.cssText = 'border:1px solid var(--yellow, #f1fa8c);border-radius:6px;padding:8px;margin-bottom:8px;';
    const heading = document.createElement('div');
    heading.style.cssText = 'font-weight:600;font-size:12px;margin-bottom:4px;';
    heading.textContent = titleText;
    card.appendChild(heading);
    return card;
  }

  function decisionBtn(label, onClick) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = label;
    b.addEventListener('click', onClick);
    return b;
  }

  function renderPendingConfirm(pending) {
    const card = actionCardShell(`Proposed action: ${pending.tool || ''}`);

    if (pending.rationale) {
      const rationale = document.createElement('div');
      rationale.style.cssText = 'font-size:11px;opacity:0.75;margin-bottom:4px;';
      rationale.textContent = pending.rationale;
      card.appendChild(rationale);
    }

    const args = document.createElement('pre');
    args.style.cssText = 'font-size:11px;opacity:0.7;white-space:pre-wrap;word-break:break-word;margin:4px 0;';
    args.textContent = JSON.stringify(pending.args || {}, null, 2);
    card.appendChild(args);

    const btnRow = document.createElement('div');
    btnRow.style.cssText = 'display:flex;gap:6px;margin-top:6px;';

    btnRow.appendChild(decisionBtn('Approve', () => sendDecision({ decision: 'approve' })));
    btnRow.appendChild(decisionBtn('Deny', () => sendDecision({ decision: 'deny' })));
    btnRow.appendChild(decisionBtn('Edit', () => {
      const raw = prompt('Edit args (JSON):', JSON.stringify(pending.args || {}));
      if (raw === null) return;
      let parsed;
      try {
        parsed = JSON.parse(raw);
      } catch (e) {
        setMsg(`Invalid JSON: ${e.message}`, true);
        return;
      }
      sendDecision({ decision: 'edit', args: parsed });
    }));
    btnRow.appendChild(decisionBtn('Stop', () => sendDecision({ decision: 'stop' })));

    card.appendChild(btnRow);
    return card;
  }

  function renderPendingAsk(pending) {
    const card = actionCardShell('Operator has a question');

    const question = document.createElement('div');
    question.style.cssText = 'font-size:12px;margin-bottom:6px;';
    question.textContent = pending.question || '';
    card.appendChild(question);

    const inputRow = document.createElement('div');
    inputRow.style.cssText = 'display:flex;gap:6px;';

    const answerInp = document.createElement('input');
    answerInp.type = 'text';
    answerInp.placeholder = 'Your answer…';
    answerInp.style.cssText = 'flex:1;';
    inputRow.appendChild(answerInp);

    // The backend writes `answer` to the pending ask regardless of the
    // `decision` string, as long as the pending kind is "ask" — we send
    // decision:"answer" for clarity but what matters is the answer field.
    inputRow.appendChild(decisionBtn('Send', () => sendDecision({ decision: 'answer', answer: answerInp.value })));
    inputRow.appendChild(decisionBtn('Stop', () => sendDecision({ decision: 'stop' })));

    card.appendChild(inputRow);
    return card;
  }

  function render(status) {
    const transcriptEl = $('operator-transcript');
    if (!transcriptEl) return;
    transcriptEl.innerHTML = '';

    const entries = Array.isArray(status.transcript) ? status.transcript : [];
    entries.forEach((entry) => transcriptEl.appendChild(renderTranscriptEntry(entry)));

    const pending = status.pending;
    if (pending) {
      transcriptEl.appendChild(pending.kind === 'ask' ? renderPendingAsk(pending) : renderPendingConfirm(pending));
    }

    if (!pending && !entries.length) {
      const empty = document.createElement('div');
      empty.className = 'admin-empty';
      empty.textContent = status.status === 'idle' ? 'No operator session running.' : 'Waiting…';
      transcriptEl.appendChild(empty);
    }
  }

  function describeResult(status, result) {
    if (!result) return `Operator ${status}`;
    if (result.error) return `Operator ${status}: ${result.error}`;
    if (typeof result.rounds === 'number') return `Operator ${status} after ${result.rounds} round(s)`;
    return `Operator ${status}`;
  }

  async function poll() {
    try {
      const res = await fetch('/api/operator/status', { credentials: 'same-origin' });
      if (!res.ok) throw new Error(String(res.status));
      const status = await res.json();
      render(status);

      const active = ACTIVE_STATUSES.includes(status.status);
      setActive(active);

      if (active) {
        if (!pollTimer) startPolling();
        return;
      }

      stopPolling();
      if (TERMINAL_STATUSES.includes(status.status) && status.status !== 'idle') {
        const isBad = status.status === 'error' || status.status === 'stuck';
        setMsg(describeResult(status.status, status.result), isBad);
      }
    } catch (e) {
      setMsg(`Status check failed: ${e.message}`, true);
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    $('tool-operator-btn')?.addEventListener('click', open);
    $('close-operator-modal')?.addEventListener('click', close);
    $('operator-modal')?.addEventListener('click', (e) => {
      if (e.target === $('operator-modal')) close();
    });
    $('operator-start-btn')?.addEventListener('click', start);
    $('operator-stop-btn')?.addEventListener('click', stop);
  });

  window.OperatorPanel = { open, close };
})();
