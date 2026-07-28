// Pure helpers for the adapter Export control — no DOM.
export const EXPORT_QUANTS = ['Q4_K_M', 'Q5_K_M', 'Q8_0', 'F16'];

export function exportButtonState(a) {
  return { canExport: !!(a && a.complete) };
}
