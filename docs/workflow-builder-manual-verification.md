# Workflow Builder — Manual Verification Runbook

The visual workflow builder shipped in five sub-projects. **108 automated tests**
cover the pure logic, validation, engine execution, JS/Python port parity, route
contracts, and tool registration. What they **cannot** cover — and what this
runbook is for — is the DOM/SVG editor UI, the live schedule/webhook firing, and
the agent invoking a workflow with a real model. Work through the checks below in
the running app; each names the expected result.

**Prerequisites**
- Run the app and sign in as an **admin** (the whole `/api/workflows` surface,
  triggers, and the `run_workflow` tool are admin-only).
- For any LLM/agent step, have a **model served** (a local model or a configured
  endpoint) so `llm` nodes and the agent actually run.
- Keep a second, **non-admin** account handy for the two security checks.

---

## 1. Visual editor (sub-project 2)

Open the **Workflows** icon on the left rail (admin only — confirm it is **hidden**
for a non-admin).

- [ ] **Build a workflow.** "+ New" → add `input` (name `q`) → `template`
  (`Q: {q}`) → `llm` (`{p}`) → `output` (name `answer`). A `{slot}` you type
  creates an input port live.
- [ ] **Wire it.** Drag from an output dot to an input dot; the wire connects. Try
  to draw a cycle — it's refused with a message.
- [ ] **Move nodes.** Drag a node by its header; its wires follow. Drag exactly to
  the top-left corner and reload later — it should NOT force a re-layout.
- [ ] **Edit config without losing focus.** Select the `template`, type into its
  field — focus holds keystroke-to-keystroke, and a new `{slot}` adds a port live.
- [ ] **Delete a wire.** Click a wire (it highlights) → press Delete/Backspace → it
  disappears. (Deleting while typing in a field must NOT delete a wire.)
- [ ] **Save / reload.** Name it, Save → it appears in the left list. Reopen it →
  **node positions are restored**.
- [ ] **Run.** "Run ▶" → answer the input prompt → the results panel shows the
  outputs and a per-node log (green `ok` pills); clicking a log row highlights that
  node. Force a failure (an `llm` node with no served model) → its row is **red**,
  downstream **grey**, and the run still returns (no crash).

## 2. Triggers (sub-project 3a)

In the editor, **Save** a workflow, then open the **Triggers** panel.

- [ ] **Save-first guard.** On an unsaved workflow the panel says "Save the workflow
  first to add triggers."
- [ ] **Schedule trigger.** Add a `daily` trigger → it lists. It also appears in the
  **Tasks** modal with run history.
- [ ] **Webhook trigger.** Add a `webhook` trigger → **Copy URL** → `POST` that URL
  with a JSON body (e.g. `{"topic":"cats"}` if the workflow has a `topic` input) →
  a run appears in Tasks, and the body values reached the workflow's inputs.
- [ ] **Event trigger.** Add a `message_sent` event trigger → send a chat message →
  after the configured count it fires, and (if the workflow has a `message` input)
  the message text is injected.
- [ ] **Delete.** Deleting a trigger removes it (and revokes its webhook token).
- [ ] **🔒 SECURITY — non-admin can't create/convert.** As a **non-admin**, `POST
  /api/tasks` with `{"task_type":"workflow","action":"<id>", ...}` → **403**. Then
  create a plain `llm` task as the non-admin and `PUT` it to `task_type=workflow`
  → **403** (the update path is gated too). Both must be refused.

## 3. Branching (sub-project 3b)

- [ ] **Add a branch.** "+ Node" → `branch` appears. Select it → the inspector shows
  a **mode** dropdown, a **cases** list (+ case / ×), and (llm mode) a **prompt** box.
- [ ] **Cases → ports.** Add cases `yes`/`no` → the node grows an output dot per case
  **plus `else`**. Rename a case → its wire drops (pruned). Delete a case → same.
- [ ] **Wire + run (match).** `input → branch.value`, and `branch.yes`/`branch.no`
  to two separate output paths. Run with an input equal to `yes` → the `yes` path is
  `ok`, the `no` path `skipped`, and only the `yes` output appears. Try a value
  matching no case → the `else` path is taken.
- [ ] **Run (llm).** Switch the branch to **llm** mode with a prompt ("does this need
  a reply?") → run → the model's choice routes the run (one path lights, the other
  greys).

## 4. Agent-callable (sub-project 3c)

With a model served, in a normal **agent** chat as an **admin**:

- [ ] **List.** Ask "what workflows can you run?" → the agent calls `run_workflow`
  (list mode) and reports the saved workflows and each one's inputs.
- [ ] **Run.** Ask "run the `<name>` workflow with topic=AI" → the agent calls
  `run_workflow` and reports the outputs.
- [ ] **🔒 SECURITY — non-admin.** As a **non-admin**, ask the agent to run a
  workflow → it must **not** be able to (the tool is admin-blocked; the agent
  doesn't have it). Confirm no workflow executes.

---

## What's already automated (don't re-check these)

- Engine: DAG execution, partial-on-failure, branch skip-cascade, input resolution.
- Validation: node/port/edge shape, cycles, branch cases, malformed-JSON → 400.
- JS↔Python **port parity** (a workflow drawn in the editor validates + runs identically).
- Route contracts: `/api/workflows` CRUD+run 400/404/200; workflow-trigger create/update admin-gating.
- Trigger input resolution (fixed < context), event payload threading, the scheduler's workflow branch.
- The `run_workflow` tool: list/run, recursion guard, never-raises, and registration at every surface.

## Known deferred (not bugs — future work)

- **Loops** and **typed ports** (not built; loops would break the engine's no-cycle invariant).

*(Closed 2a01911: the non-dict-`config` hardening — a malformed non-dict `config` on any node
type now returns a 400 "config must be an object" error instead of a 500. `validate` no longer
raises on any untrusted-JSON shape.)*
