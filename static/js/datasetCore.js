// Pure helpers for the Dataset builder — no DOM. Mirrors the training row shapes.
export const ROW_FORMATS = {
  text: ['text'],
  instruction: ['instruction', 'input', 'response'],
  prompt: ['prompt', 'completion'],
};

export function formToRow(format, fields) {
  const f = (fields && typeof fields === 'object') ? fields : {};
  // Coerce truthy non-strings to string so a stray number/array can't throw
  // (.trim is string-only); falsy values stay '' as before.
  const g = function (k) { const v = f[k]; return (v ? String(v) : '').trim(); };
  if (format === 'text') {
    return g('text') ? { row: { text: g('text') }, error: null } : { row: null, error: 'text is required' };
  }
  if (format === 'instruction') {
    if (!g('instruction')) return { row: null, error: 'instruction is required' };
    if (!g('response')) return { row: null, error: 'response is required' };
    const row = { instruction: g('instruction'), response: g('response') };
    if (g('input')) row.input = g('input');
    return { row: row, error: null };
  }
  if (format === 'prompt') {
    if (!g('prompt')) return { row: null, error: 'prompt is required' };
    if (!g('completion')) return { row: null, error: 'completion is required' };
    return { row: { prompt: g('prompt'), completion: g('completion') }, error: null };
  }
  return { row: null, error: 'unknown format' };
}
