# Training GUI — Manual Verification Runbook

The Training panel has no automated UI test (the run form drives real GPU work).
After a frozen rebuild, an admin verifies it by hand.

## Prerequisites
- `python scripts/fetch_uv.py` has vendored `uv` into `build_assets/uv/`, and the
  app was rebuilt (clean) so `training_sidecar/` + `uv` are bundled.
- Logged in as an admin.

## Steps
1. **Rail button (admin gate):** the Training icon appears in the sidebar icon-rail.
   Log in as a non-admin (or hit `/api/auth/status` returning `is_admin:false`) →
   the button is hidden.
2. **Open the panel:** click the Training icon → the modal opens with an
   Environment card, a run form, and an Adapters list.
3. **Env setup:** if status is `not_installed`, click **Set up training
   environment**. It runs the one-time (~3–4 GB) install; on success the status
   flips to `ready` and the setup button hides. (First run only; needs internet.)
4. **VRAM hint:** type `Qwen/Qwen2.5-0.5B-Instruct` in the base-model field → the
   hint reads roughly `~1.4 GB of <free> GB — fits`. Type a `13B` model → `likely
   too big`.
5. **Start a run:** set a `.jsonl` dataset path (a few `{"text": …}` lines),
   keep Steps small (e.g. 20), click **Start training**. The progress line updates
   live (`status: running · step N · loss … · vram … GB`).
6. **Completion:** on finish, status shows `done` and the run appears in the
   **Adapters** list with ✅ and the base model. Confirm the adapter files exist
   under `<DATA_DIR>/training/adapters/<run-id>/`.
7. **Stop:** start another run and click **Stop** → status shows `stopped` (not
   `error`).

## Owed regressions to watch
- Live `step` events must reach the progress line (guards the `disable_tqdm`
  fix — a tqdm bar on the JSON channel would blank the telemetry).
- A user **Stop** must read `stopped`, never `error`.
- Closing the panel (✕) or reloading the page **during** an active run and
  reopening it must resume the live progress line (guards the poll-resume fix).

## Serving a tuned adapter (sub-project 3)
- After a completed run, the adapter row shows **Convert to GGUF**; clicking it
  writes `adapter.gguf` next to the adapter and the row then shows **Serve**.
- **Serve** auto-fills the matched base GGUF (or prompts to pick/download one);
  confirm and serve. The tuned model appears in the model selector as
  `<base> · <run-id> (LoRA)` and its output differs from the base.
- A wrong/mismatched base must fail with a clear error (surfaced from the
  `llama-server` log tail), never a crash.
