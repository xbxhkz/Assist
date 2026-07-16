# EXL2 / MLX Models via External Servers — Design

**Goal:** Make EXL2 and MLX models first-class *connectable* models in Assist by
adding one-click endpoint presets for their OpenAI-compatible servers (TabbyAPI
for EXL2, `mlx_lm.server` for MLX) plus honest setup guidance — reusing the
existing model-endpoint system, with no bundled GPU-vendor runtimes.

**Scope:** Frontend presets + setup docs + a browsing hint. **No native local
serving** of EXL2/MLX (deliberately out of scope — see Rationale). No backend
changes.

---

## Rationale — why external servers, not native serving

Local first-party serving of these formats is infeasible/undesirable on the
target hardware:

- **MLX is Apple-Silicon only** (`mlx_lm` cannot run on Windows/NVIDIA at all).
- **EXL2 (exllamav2) is CUDA-only** and needs a CUDA build of torch (~2.5 GB;
  the app bundles CPU-only `torch 2.11.0+cpu`, shared by Whisper/Kokoro/YOLO)
  plus a compiled exllamav2 extension — a large, fragile, per-CUDA-version
  bundle for marginal gain over the existing GGUF fill-and-spill path on 6 GB
  VRAM.

The codebase already made this call: `services/hwfit/fit.py` **recognizes**
EXL2/MLX/AWQ/GPTQ for browsing and fit-scoring but explicitly *"does not generate
serve commands"* for MLX. Meanwhile Assist already connects to **any
OpenAI-compatible endpoint** (`ModelEndpoint`, `core/database.py:361`), and both
TabbyAPI (EXL2) and `mlx_lm.server` (MLX) expose OpenAI-compatible APIs. So the
valuable, feasible feature is to make *connecting to those servers* a
first-class, guided experience.

## Background — what exists (and is reused)

- **Endpoint system** — `ModelEndpoint` (`core/database.py:361`); admin "Add a
  local model server" card in `static/index.html` (~line 2510) with an input
  `adm-epLocalUrl`, an Add button `adm-epLocalAddBtn` (`POST
  /api/model-endpoints`), and a preset "more" menu `adm-epLocalMoreMenu`
  containing **"Add Ollama"** (`adm-epOllamaBtn`) and "Scan network". The Ollama
  button (`static/js/admin.js:~1622`) simply fills `adm-epLocalUrl` with the
  default Ollama URL — the exact pattern the new presets mirror.
- Once an endpoint is added, the existing probe (`/api/model-endpoints/.../models`,
  online-status) lists its models and they appear in the model picker like any
  other endpoint model — nothing new needed downstream.
- **Format recognition** — `services/hwfit/models.py` / `fit.py` detect
  EXL2/MLX; `static/js/cookbook.js` already renders such repos when browsing.
- **Help ▸ Manual** — an accordion of `<details>` sections in
  `static/index.html` (`id="help-tab-manual"`).

## Components

### 1. Endpoint presets (`static/index.html` + `static/js/admin.js`)

Add two buttons to `adm-epLocalMoreMenu`, mirroring `adm-epOllamaBtn`:

- **"Add TabbyAPI (EXL2)"** → fills `adm-epLocalUrl` with
  `http://127.0.0.1:5000/v1` (TabbyAPI's default host/port).
- **"Add MLX (mlx_lm)"** → fills with `http://127.0.0.1:8080/v1`
  (`mlx_lm.server`'s default port).

Each click also sets the type select to `llm` and shows a one-line inline
message in `adm-epLocalMsg` (e.g. *"TabbyAPI URL filled — start the server, then
click Add. See Help ▸ Manual for setup."*). The two default URLs live as named
constants in `admin.js` (`TABBY_DEFAULT_URL`, `MLX_DEFAULT_URL`) so they can be
unit-pinned. The card sub-text (`static/index.html` ~line 2519) updates to
mention *TabbyAPI/EXL2* and *MLX*.

### 2. Setup guidance (`static/index.html`, Help ▸ Manual)

A new `<details>` section **"EXL2 & MLX models (external servers)"** with exact,
copy-pasteable steps and honest platform notes:

- **EXL2 — NVIDIA/CUDA (Windows/Linux):** install **TabbyAPI**
  (`github.com/theroyallab/tabbyAPI`), run its `start.bat` (Windows) /
  `start.sh`, put an `.exl2`/`exl2` model in its `models/` folder → it serves an
  OpenAI-compatible API at `http://127.0.0.1:5000/v1`. Then use the **Add
  TabbyAPI (EXL2)** preset and click **Add**. Requires an NVIDIA GPU + CUDA.
- **MLX — Apple Silicon only:** `pip install mlx-lm`, then
  `mlx_lm.server --model <hf-repo-or-path> --port 8080` → serves at
  `http://127.0.0.1:8080/v1`. Use the **Add MLX (mlx_lm)** preset → **Add**.
  Explicitly notes **macOS / Apple-Silicon only** (won't run on Windows/NVIDIA).
- One line noting that once added, the server's models appear in the model
  picker like any other endpoint.

### 3. Cookbook format hint (`static/js/cookbook.js`)

When the model browser renders a repo detected as EXL2 or MLX, show a small,
non-blocking note — *"EXL2/MLX: serve via an external endpoint (TabbyAPI / MLX)
— see Settings → Add model endpoints"* — instead of implying a local GGUF-style
serve. Reuses the existing format detection (the `mlx`/`exl2` checks already in
`cookbook.js`/`hwfit`); CSP-safe (`textContent`, no `innerHTML`-with-data).

## Data flow

```
User runs TabbyAPI / mlx_lm.server (their machine)
   → clicks preset (fills adm-epLocalUrl with the default base URL)
   → Add  → POST /api/model-endpoints (existing)
   → existing probe lists the server's models
   → EXL2/MLX model appears in the model picker like any endpoint model
```

## Error handling

Nothing new. An unstarted server → the endpoint shows **offline** (existing
probe behavior); the guidance says to start the server first. A wrong URL is the
user's to correct via the same existing flow.

## Testing

- **Unit (node-from-pytest, the repo pattern):** if the two preset URLs are
  exported as constants from a small helper, a test pins
  `TABBY_DEFAULT_URL === "http://127.0.0.1:5000/v1"` and
  `MLX_DEFAULT_URL === "http://127.0.0.1:8080/v1"`. Otherwise the constants are
  asserted by a grep-style test. No backend changes → no backend tests.
- **Live-verify:** open Settings → Add model endpoints → the "…" menu shows
  **Add TabbyAPI (EXL2)** and **Add MLX (mlx_lm)**; each fills the correct URL +
  inline hint; the Help ▸ Manual section renders with the setup steps; the
  Cookbook shows the external-endpoint hint on an EXL2/MLX repo. (Actually
  connecting to a running TabbyAPI is optional — the endpoint system itself is
  pre-existing and already tested.)

## Non-goals

- Native local serving of EXL2 or MLX (no bundled exllamav2/CUDA-torch/mlx).
- Auto-starting or installing TabbyAPI/`mlx_lm` for the user.
- Downloading EXL2/MLX weights through the app's serve pipeline (browsing/download
  already works; serving is the external server's job).
- AWQ/GPTQ/other formats (same external-endpoint path applies, but not in scope).
