// Pure helpers for the Training panel — no DOM, no fetch. Mirrors
// src/training/config.py (estimate_vram_gb / fit_level / parse_params_b) so the
// client-side hint matches the server's soft VRAM gate.
export function parseParamsB(modelId) {
  if (typeof modelId !== 'string') return null;
  const m = modelId.replace(/_/g, '-').match(/(\d+(?:\.\d+)?)\s*[bB]\b/);
  return m ? parseFloat(m[1]) : null;
}

const FIXED = 1.0, PER_B = 0.8;
export function estimateVramGb(paramsB) {
  const pb = Number(paramsB);
  if (!isFinite(pb)) return FIXED;
  return Math.round((FIXED + PER_B * Math.max(pb, 0)) * 100) / 100;
}

export function vramHint(paramsB, freeGib) {
  if (freeGib == null || paramsB == null) return { level: 'unknown', label: 'VRAM: unknown' };
  const est = estimateVramGb(paramsB);
  let level = 'too_big';
  if (est <= 0.8 * freeGib) level = 'fits';
  else if (est <= freeGib) level = 'tight';
  const note = level === 'fits' ? 'fits' : level === 'tight' ? 'tight — may OOM' : 'likely too big';
  return { level, label: '~' + est + ' GB of ' + freeGib + ' GB — ' + note };
}

export function formToConfig(v) {
  const cfg = {
    base_model: (v.base_model || '').trim(),
    dataset_path: (v.dataset_path || '').trim(),
    lora_r: parseInt(v.lora_r, 10) || 8,
    lora_alpha: parseInt(v.lora_alpha, 10) || 16,
    lora_dropout: (v.lora_dropout != null && v.lora_dropout !== '') ? parseFloat(v.lora_dropout) : 0.05,
    batch_size: parseInt(v.batch_size, 10) || 1,
    learning_rate: parseFloat(v.learning_rate) || 2e-4,
    max_seq_length: parseInt(v.max_seq_length, 10) || 512,
  };
  if (v.mode === 'epochs') { cfg.epochs = parseFloat(v.epochs) || 1; cfg.steps = null; }
  else { cfg.steps = parseInt(v.steps, 10) || 100; cfg.epochs = null; }
  return cfg;
}
