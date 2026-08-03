// Pure helpers for the Image Training panel -- no DOM, no fetch. Mirrors
// static/js/trainingCore.js's shape for the SDXL image-LoRA training engine
// (src/image_training/config.py's ImageTrainingConfig fields). base_model is
// intentionally never produced here -- the server already defaults it to the
// one supported value (SUPPORTED_BASE_MODELS is a single-entry allowlist).
export function formToConfig(v) {
  return {
    dataset_name: (v.dataset_name || '').trim(),
    output_name: (v.output_name || '').trim(),
    rank: parseInt(v.rank, 10) || 4,
    lora_alpha: parseInt(v.lora_alpha, 10) || 4,
    learning_rate: parseFloat(v.learning_rate) || 1e-4,
    steps: parseInt(v.steps, 10) || 1000,
    resolution: parseInt(v.resolution, 10) || 1024,
  };
}

export function renderStatusLine(s) {
  const bits = ['status: ' + s.status];
  if (s.last_step != null) bits.push('step ' + s.last_step);
  if (s.loss != null) bits.push('loss ' + s.loss);
  if (s.vram_gb != null) bits.push('vram ' + s.vram_gb + ' GB');
  if (s.status === 'done' && s.lora_path != null) bits.push('saved: ' + s.lora_path);
  if (s.error) bits.push('(' + s.error + ')');
  return bits.join(' · ');
}
