/**
 * Face Swap wiring — the one Gallery editor AI action that needs a
 * SECOND uploaded file (the source face) alongside the flattened
 * editor canvas (the target). No other AI tool in this editor takes a
 * secondary upload, so this file — unlike ai-rembg.js / ai-tools-misc.js
 * — owns its own hidden `<input type="file">` (same pattern as
 * editor/wire-import.js's importFileInput) rather than reusing
 * applyImageTool's single-image JSON contract.
 *
 *   ge-faceswap-choose:  opens the hidden file picker for the source
 *     face photo. On selection, shows a filename label + thumbnail
 *     preview and enables the Swap Face button.
 *
 *   ge-faceswap-run:  flattens the canvas (target) and POSTs both
 *     images as multipart/form-data to /api/image/face-swap —
 *     mirroring Style Transfer's request shape in ai-tools-misc.js
 *     (`fd.append('image', blob, ...)`) plus a second field,
 *     `source_face`, for the uploaded file. The result lands as a new
 *     "Face Swapped" layer, same as every other AI tool here.
 *
 * @param {{
 *   apiBase:           string,
 *   container:         HTMLElement,
 *   flatten:            () => HTMLCanvasElement,
 *   saveState:          (label?: string) => void,
 *   createLayer:        (name: string, w: number, h: number) => object,
 *   composite:          () => void,
 *   renderLayerPanel:   () => void,
 *   uiModule:           object,
 * }} deps
 */
import { state } from './state.js';

export function wireFaceSwap({
  apiBase, container, flatten, saveState, createLayer, composite,
  renderLayerPanel, uiModule,
}) {
  let sourceFaceFile = null;

  // Hidden <input type="file"> the "Choose Source Face…" button clicks —
  // same pattern as wire-import.js's importFileInput.
  const sourceInput = document.createElement('input');
  sourceInput.type = 'file';
  sourceInput.accept = 'image/*';
  sourceInput.style.display = 'none';
  container.appendChild(sourceInput);

  const chooseBtn = document.getElementById('ge-faceswap-choose');
  const filenameLabel = document.getElementById('ge-faceswap-filename');
  const previewWrap = document.getElementById('ge-faceswap-preview-wrap');
  const previewImg = document.getElementById('ge-faceswap-preview');
  const runBtn = document.getElementById('ge-faceswap-run');

  chooseBtn?.addEventListener('click', () => sourceInput.click());

  sourceInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    sourceFaceFile = file;
    if (filenameLabel) filenameLabel.textContent = file.name;
    if (previewImg && previewWrap) {
      const url = URL.createObjectURL(file);
      previewImg.onload = () => URL.revokeObjectURL(url);
      previewImg.src = url;
      previewWrap.style.display = '';
    }
    if (runBtn) runBtn.disabled = false;
  });

  runBtn?.addEventListener('click', async () => {
    if (!sourceFaceFile) {
      if (uiModule) uiModule.showToast('Choose a source face photo first');
      return;
    }
    const btn = runBtn;
    const origHTML = btn.innerHTML;
    btn.disabled = true;
    btn.textContent = 'Swapping…';
    try {
      const flat = flatten();
      const blob = await new Promise(r => flat.toBlob(r, 'image/png'));
      const fd = new FormData();
      fd.append('image', blob, 'target.png');
      fd.append('source_face', sourceFaceFile, sourceFaceFile.name);
      const res = await fetch(`${apiBase}/api/image/face-swap`, {
        method: 'POST', credentials: 'same-origin', body: fd,
      });
      if (!res.ok) throw new Error('Server returned ' + res.status);
      const data = await res.json();
      if (data.image) {
        const img = new Image();
        img.onload = () => {
          if (!state.editorOpen) return;
          saveState();
          const layer = createLayer('Face Swapped', state.imgWidth, state.imgHeight);
          layer.ctx.drawImage(img, 0, 0, state.imgWidth, state.imgHeight);
          state.layers.push(layer);
          state.activeLayerId = layer.id;
          composite();
          renderLayerPanel();
          if (uiModule) uiModule.showToast('Face swap complete');
        };
        img.src = 'data:image/png;base64,' + data.image;
      } else {
        throw new Error(data.error || 'No image returned');
      }
    } catch (e) {
      if (uiModule) uiModule.showToast('Face swap failed: ' + e.message);
    }
    btn.disabled = false;
    btn.innerHTML = origHTML;
  });
}
