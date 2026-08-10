// static/js/toolCallLog.js
import * as Modals from './modalManager.js';

function $(id) { return document.getElementById(id); }

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
  });
}

async function api(path) {
  const res = await fetch(path, { credentials: 'same-origin' });
  const data = await res.json().catch(function () { return {}; });
  if (!res.ok) {
    const d = data && data.detail;
    throw new Error(typeof d === 'string' ? d : (res.statusText || String(res.status)));
  }
  return data;
}

let _offset = 0;
const _PAGE_SIZE = 50;
let _toolFilter = '';
let _sessionFilter = '';

function _renderEntries(entries, append) {
  const list = $('tool-call-log-list');
  if (!list) return;
  const html = entries.map(function (c) {
    const cmd = esc(c.command || '');
    const output = esc((c.output || '').slice(0, 200));
    const exitCode = (c.exit_code === null || c.exit_code === undefined) ? '' :
      ' <span class="tool-call-log-exit">exit ' + esc(c.exit_code) + '</span>';
    const sid = esc(c.session_id || '');
    return (
      '<div class="tool-call-log-entry">' +
      '<div class="tool-call-log-meta">' +
      '<a href="#" class="tool-call-log-session" data-session-id="' + sid + '">' + esc(c.session_name || 'Unknown') + '</a>' +
      ' (<a href="#" class="tool-call-log-session-filter" data-session-id="' + sid + '">filter</a>)' +
      ' &middot; ' + esc(c.tool || '?') + exitCode +
      ' &middot; <span class="tool-call-log-time">' + esc(c.timestamp || '') + '</span>' +
      '</div>' +
      '<div class="tool-call-log-command">' + cmd + '</div>' +
      '<div class="tool-call-log-output">' + output + '</div>' +
      '</div>'
    );
  }).join('');
  if (append) {
    list.insertAdjacentHTML('beforeend', html);
  } else {
    list.innerHTML = html || '<div>No tool calls yet</div>';
  }
}

async function loadToolCallLog(append) {
  const list = $('tool-call-log-list');
  if (list && !append) list.classList.remove('tool-call-log-error');
  const moreBtn = $('tool-call-log-more');
  try {
    const params = new URLSearchParams();
    params.set('limit', String(_PAGE_SIZE));
    params.set('offset', String(append ? _offset : 0));
    if (_sessionFilter) params.set('session_id', _sessionFilter);
    if (_toolFilter) params.set('tool_name', _toolFilter);
    const data = await api('/api/tool-calls?' + params.toString());
    const entries = data.tool_calls || [];
    _renderEntries(entries, append);
    _offset = (append ? _offset : 0) + entries.length;
    if (moreBtn) moreBtn.style.display = data.has_more ? '' : 'none';
  } catch (e) {
    if (list) {
      list.classList.add('tool-call-log-error');
      list.textContent = 'Failed to load: ' + e.message;
    }
  }
}

function openToolCallLog() {
  $('tool-call-log-modal').classList.remove('hidden');
  _offset = 0;
  loadToolCallLog(false);
}

function closeToolCallLog() {
  $('tool-call-log-modal').classList.add('hidden');
}

function init() {
  const rail = $('rail-tool-calls');
  if (rail) rail.addEventListener('click', openToolCallLog);
  const side = $('tool-tool-calls-btn');
  if (side) side.addEventListener('click', openToolCallLog);
  const x = $('tool-call-log-close');
  if (x) x.addEventListener('click', closeToolCallLog);

  const applyBtn = $('tool-call-log-apply-filter');
  if (applyBtn) applyBtn.addEventListener('click', function () {
    const toolInput = $('tool-call-log-tool-filter');
    const sessionInput = $('tool-call-log-session-filter-input');
    _toolFilter = toolInput ? toolInput.value.trim() : '';
    _sessionFilter = sessionInput ? sessionInput.value.trim() : '';
    _offset = 0;
    loadToolCallLog(false);
  });

  const clearBtn = $('tool-call-log-clear-filter');
  if (clearBtn) clearBtn.addEventListener('click', function () {
    _toolFilter = '';
    _sessionFilter = '';
    const toolInput = $('tool-call-log-tool-filter');
    const sessionInput = $('tool-call-log-session-filter-input');
    if (toolInput) toolInput.value = '';
    if (sessionInput) sessionInput.value = '';
    _offset = 0;
    loadToolCallLog(false);
  });

  const moreBtn = $('tool-call-log-more');
  if (moreBtn) moreBtn.addEventListener('click', function () {
    loadToolCallLog(true);
  });

  const list = $('tool-call-log-list');
  if (list) list.addEventListener('click', function (ev) {
    const jump = ev.target.closest('.tool-call-log-session');
    if (jump) {
      ev.preventDefault();
      const sid = jump.getAttribute('data-session-id');
      if (sid) {
        closeToolCallLog();
        if (window.sessionModule && window.sessionModule.selectSession) {
          window.sessionModule.selectSession(sid);
        }
      }
      return;
    }
    const filterLink = ev.target.closest('.tool-call-log-session-filter');
    if (filterLink) {
      ev.preventDefault();
      const sid = filterLink.getAttribute('data-session-id');
      if (sid) {
        _sessionFilter = sid;
        const sessionInput = $('tool-call-log-session-filter-input');
        if (sessionInput) sessionInput.value = sid;
        _offset = 0;
        loadToolCallLog(false);
      }
    }
  });

  Modals.register('tool-call-log-modal', {
    railBtnId: 'rail-tool-calls', sidebarBtnId: 'tool-tool-calls-btn', closeFn: closeToolCallLog,
  });
}

document.addEventListener('DOMContentLoaded', init);
