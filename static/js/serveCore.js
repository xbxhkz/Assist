// Pure helper for the adapter Convert/Serve button states — no DOM.
export function adapterActions(a) {
  const complete = !!(a && a.complete);
  const converted = !!(a && a.converted);
  return { canConvert: complete && !converted, canServe: complete && converted };
}
